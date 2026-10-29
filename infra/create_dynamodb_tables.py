#!/usr/bin/env python3
"""Create Shibu AI's DynamoDB tables (Step 2 + Step 3 of the DynamoDB plan).

Local-only infrastructure script. It does NOT touch GitHub and does NOT
push anything anywhere. It only talks to AWS DynamoDB in the region you
pass, using credentials from boto3's normal credential chain
(environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
~/.aws/credentials, or an IAM role). No secret is stored in this file.

Approved design (2026-10-29, Kartik approved, GitHub untouched):
  * Region default: ap-south-1 (Mumbai). Change with --region before
    you run it the first time; changing it later means new tables.
  * Billing: PROVISIONED, capped so the total stays inside the
    DynamoDB always-free allowance (25 RCU / 25 WCU per region).
    Total here: 21 RCU / 21 WCU (see TABLE_SPECS).
  * Encryption at rest: AWS-owned key (AES256, $0, DynamoDB default).
  * Point-in-time recovery: ON by default (Step 6 requirement).
    It is NOT free: about $0.20/GB-month, so cents at Shibu's size.
    Pass --no-pitr to stay strictly free.
  * TTL: enabled only on shibu-cached-answers, attribute "expires_at".
    The app (Step 4) sets expires_at = now + 30 days when it caches
    a web answer. DynamoDB then deletes the item automatically,
    usually within a day or two after expiry, not to the minute.
  * Tables (prefix can be changed with --prefix, default "shibu"):
      shibu-users         PK user_id (google#<sub>)
      shibu-conversations  PK user_id, SK conversation_id
                           GSI by-updated (user_id, updated_at)
      shibu-messages       PK conversation_id, SK created_at_id
                             (value: "<ISO timestamp>#<message_id>")
                           GSI learning-index (learning_index_key,
                             rated_at_id) -- sparse: only messages
                             approved for learning have this key
      shibu-cached-answers PK query_hash (SHA-256 of normalised query)
                             shared cache, NO user_id stored

Usage (after creating the AWS account / IAM user in AWS_SETUP.md):
    pip install -r infra/requirements.txt
    python infra/create_dynamodb_tables.py --region ap-south-1
    python infra/create_dynamodb_tables.py --region ap-south-1 --dry-run
    python infra/create_dynamodb_tables.py --region ap-south-1 --no-pitr

Re-running the script is safe: existing tables are left alone
(idempotent), and TTL / PITR are re-applied if missing.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_REGION = "ap-south-1"
DEFAULT_PREFIX = "shibu"

# Each spec is pure data, so tests can import it without boto3.
TABLE_SPECS = [
    {
        "suffix": "users",
        "key_schema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
        "attributes": [{"AttributeName": "user_id", "AttributeType": "S"}],
        "read_units": 2,
        "write_units": 2,
        "gsis": [],
        "ttl_attribute": None,
    },
    {
        "suffix": "conversations",
        "key_schema": [
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "conversation_id", "KeyType": "RANGE"},
        ],
        "attributes": [
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "conversation_id", "AttributeType": "S"},
            {"AttributeName": "updated_at", "AttributeType": "S"},
        ],
        "read_units": 3,
        "write_units": 3,
        "gsis": [
            {
                "IndexName": "by-updated",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "updated_at", "KeyType": "RANGE"},
                ],
                "Projection": {
                    "ProjectionType": "INCLUDE",
                    "NonKeyAttributes": ["title", "summary"],
                },
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 3,
                    "WriteCapacityUnits": 3,
                },
            }
        ],
        "ttl_attribute": None,
    },
    {
        "suffix": "messages",
        "key_schema": [
            {"AttributeName": "conversation_id", "KeyType": "HASH"},
            {"AttributeName": "created_at_id", "KeyType": "RANGE"},
        ],
        "attributes": [
            {"AttributeName": "conversation_id", "AttributeType": "S"},
            {"AttributeName": "created_at_id", "AttributeType": "S"},
            {"AttributeName": "learning_index_key", "AttributeType": "S"},
            {"AttributeName": "rated_at_id", "AttributeType": "S"},
        ],
        "read_units": 8,
        "write_units": 8,
        "gsis": [
            {
                "IndexName": "learning-index",
                "KeySchema": [
                    {"AttributeName": "learning_index_key", "KeyType": "HASH"},
                    {"AttributeName": "rated_at_id", "KeyType": "RANGE"},
                ],
                # KEYS_ONLY keeps the GSI small: the nightly job
                # batch-reads the full messages it needs.
                "Projection": {"ProjectionType": "KEYS_ONLY"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 2,
                    "WriteCapacityUnits": 2,
                },
            }
        ],
        "ttl_attribute": None,
    },
    {
        "suffix": "cached-answers",
        "key_schema": [{"AttributeName": "query_hash", "KeyType": "HASH"}],
        "attributes": [{"AttributeName": "query_hash", "AttributeType": "S"}],
        "read_units": 3,
        "write_units": 3,
        "gsis": [],
        "ttl_attribute": "expires_at",
    },
]


def table_name(prefix: str, suffix: str) -> str:
    return f"{prefix}-{suffix}"


def total_capacity() -> tuple[int, int]:
    """Total provisioned RCU / WCU across tables + GSIs."""
    reads = writes = 0
    for spec in TABLE_SPECS:
        reads += spec["read_units"]
        writes += spec["write_units"]
        for gsi in spec["gsis"]:
            reads += gsi["ProvisionedThroughput"]["ReadCapacityUnits"]
            writes += gsi["ProvisionedThroughput"]["WriteCapacityUnits"]
    return reads, writes


def build_create_kwargs(prefix: str, spec: dict) -> dict:
    kwargs = {
        "TableName": table_name(prefix, spec["suffix"]),
        "KeySchema": spec["key_schema"],
        "AttributeDefinitions": spec["attributes"],
        "BillingMode": "PROVISIONED",
        "ProvisionedThroughput": {
            "ReadCapacityUnits": spec["read_units"],
            "WriteCapacityUnits": spec["write_units"],
        },
        # AWS-owned key: encryption at rest, $0, cannot be disabled.
        "SSESpecification": {"Enabled": True, "SSEType": "AES256"},
        "Tags": [
            {"Key": "Project", "Value": "shibu-ai"},
            {"Key": "ManagedBy", "Value": "create_dynamodb_tables.py"},
        ],
    }
    if spec["gsis"]:
        kwargs["GlobalSecondaryIndexes"] = spec["gsis"]
    return kwargs


def table_exists(client, name: str) -> bool:
    try:
        client.describe_table(TableName=name)
        return True
    except client.exceptions.ResourceNotFoundException:
        return False
    except Exception as exc:  # moto / older clients surface it differently
        if "ResourceNotFound" in str(exc):
            return False
        raise


def ensure_ttl(client, name: str, attribute: str) -> None:
    try:
        client.update_time_to_live(
            TableName=name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": attribute},
        )
        print(f"  TTL enabled on {name} (attribute: {attribute})")
    except Exception as exc:
        print(f"  WARNING: could not enable TTL on {name}: {exc}")


def ensure_pitr(client, name: str) -> None:
    try:
        client.update_continuous_backups(
            TableName=name,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )
        print(f"  PITR enabled on {name}")
    except Exception as exc:
        print(f"  WARNING: could not enable PITR on {name}: {exc}")


def create_all_tables(region: str, prefix: str, enable_pitr: bool,
                      dry_run: bool = False) -> list[str]:
    reads, writes = total_capacity()
    print(f"Region: {region} | Prefix: {prefix}")
    print(f"Total provisioned capacity: {reads} RCU / {writes} WCU "
          f"(free allowance is 25 / 25)")
    if reads > 25 or writes > 25:
        raise SystemExit("ERROR: capacity exceeds the 25/25 free allowance.")

    created_or_existing = []
    if dry_run:
        for spec in TABLE_SPECS:
            kwargs = build_create_kwargs(prefix, spec)
            print(f"  [dry-run] would create {kwargs['TableName']} "
                  f"({spec['read_units']} RCU / {spec['write_units']} WCU, "
                  f"GSIs: {[g['IndexName'] for g in spec['gsis']]}, "
                  f"TTL: {spec['ttl_attribute']})")
            created_or_existing.append(kwargs["TableName"])
        return created_or_existing

    import boto3  # imported here so spec tests do not need boto3

    client = boto3.client("dynamodb", region_name=region)
    for spec in TABLE_SPECS:
        kwargs = build_create_kwargs(prefix, spec)
        name = kwargs["TableName"]
        if table_exists(client, name):
            print(f"  {name} already exists, leaving it alone")
        else:
            print(f"  Creating {name} ...")
            client.create_table(**kwargs)
            client.get_waiter("table_exists").wait(TableName=name)
            print(f"  {name} is ACTIVE")
        if spec["ttl_attribute"]:
            ensure_ttl(client, name, spec["ttl_attribute"])
        if enable_pitr:
            ensure_pitr(client, name)
        created_or_existing.append(name)
    return created_or_existing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Create Shibu AI's DynamoDB tables (local script, "
                    "no GitHub changes).")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help="AWS region, default ap-south-1 (Mumbai)")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="Table name prefix, default shibu")
    parser.add_argument("--no-pitr", action="store_true",
                        help="Skip point-in-time recovery (strictly free, "
                             "but no continuous backups)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created, call no AWS APIs")
    args = parser.parse_args(argv)
    try:
        names = create_all_tables(
            region=args.region, prefix=args.prefix,
            enable_pitr=not args.no_pitr, dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Done. Tables: " + ", ".join(names))
    print("Verify in AWS Console > DynamoDB > Tables, same region.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
