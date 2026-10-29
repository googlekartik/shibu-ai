"""Step 7: admin usage page data for DynamoDB.

IMPORTANT LIMITATION, stated on the page itself: DynamoDB does
not report "free-tier used so far" directly. This page therefore
approximates from DescribeTable (provisioned RCU/WCU, item count,
table size in bytes) plus, when CloudWatch is available,
ConsumedRead/WriteCapacityUnits over the last 24 hours. It is an
approximation, not a billing statement — the AWS Billing console
is the source of truth for charges.

Free-tier reference (provisioned mode, per region, per account):
  25 RCU, 25 WCU, 25 GB storage. Shibu is designed for 21/21,
  so 80% of the free limit is 20 RCU / 20 WCU / 20 GB.
A warning is shown at >=80% of any of those, and the page says
"approximate" next to every number that is not a direct table
property.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

FREE_RCU = 25
FREE_WCU = 25
FREE_STORAGE_GB = 25
WARNING_PCT = 80
TABLES = ["shibu-users", "shibu-conversations", "shibu-messages", "shibu-cached-answers"]


def _pct(used, limit):
    return round(100 * used / limit, 1) if limit else 0.0


def get_usage(region: str = "ap-south-1", prefix: str = "shibu",
              dynamodb_client=None, cloudwatch_client=None) -> dict:
    """Return usage dict. Clients can be injected for tests (moto)."""
    if dynamodb_client is None:
        import boto3

        dynamodb_client = boto3.client("dynamodb", region_name=region)
    tables = []
    total_rcu = 0
    total_wcu = 0
    total_bytes = 0
    total_items = 0
    for short in ("users", "conversations", "messages", "cached-answers"):
        name = f"{prefix}-{short}"
        try:
            desc = dynamodb_client.describe_table(TableName=name)["Table"]
        except Exception as exc:
            tables.append({"table": name, "status": "error", "error": str(exc)})
            continue
        throughput = desc.get("ProvisionedThroughput", {})
        rcu = int(throughput.get("ReadCapacityUnits", 0))
        wcu = int(throughput.get("WriteCapacityUnits", 0))
        # GSIs also consume provisioned capacity.
        for gsi in desc.get("GlobalSecondaryIndexes", []) or []:
            gt = gsi.get("ProvisionedThroughput", {})
            rcu += int(gt.get("ReadCapacityUnits", 0))
            wcu += int(gt.get("WriteCapacityUnits", 0))
        size = int(desc.get("TableSizeBytes", 0))
        items = int(desc.get("ItemCount", 0))
        total_rcu += rcu
        total_wcu += wcu
        total_bytes += size
        total_items += items
        tables.append({
            "table": name, "status": desc.get("TableStatus", ""),
            "provisioned_rcu": rcu, "provisioned_wcu": wcu,
            "item_count": items, "size_bytes": size,
            "size_gb": round(size / (1024 ** 3), 4),
        })

    # CloudWatch consumed capacity (last 24h, best effort only).
    consumed = {"rcu_24h": None, "wcu_24h": None, "cloudwatch_available": False}
    if cloudwatch_client is None:
        try:
            import boto3

            cloudwatch_client = boto3.client("cloudwatch", region_name=region)
        except Exception:
            cloudwatch_client = None
    if cloudwatch_client is not None:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=24)
            for metric, key in (("ConsumedReadCapacityUnits", "rcu_24h"),
                                ("ConsumedWriteCapacityUnits", "wcu_24h")):
                total = 0.0
                for name in [f"{prefix}-{s}" for s in
                             ("users", "conversations", "messages", "cached-answers")]:
                    resp = cloudwatch_client.get_metric_statistics(
                        Namespace="AWS/DynamoDB", MetricName=metric,
                        Dimensions=[{"Name": "TableName", "Value": name}],
                        StartTime=start, EndTime=end, Period=86400,
                        Statistics=["Sum"])
                    for point in resp.get("Datapoints", []):
                        total += float(point.get("Sum", 0))
                consumed[key] = total
            consumed["cloudwatch_available"] = True
        except Exception:
            consumed["cloudwatch_available"] = False

    storage_gb = round(total_bytes / (1024 ** 3), 4)
    warnings = []
    if _pct(total_rcu, FREE_RCU) >= WARNING_PCT:
        warnings.append(f"Provisioned read capacity {total_rcu}/{FREE_RCU} RCU is at or above 80% of the free limit.")
    if _pct(total_wcu, FREE_WCU) >= WARNING_PCT:
        warnings.append(f"Provisioned write capacity {total_wcu}/{FREE_WCU} WCU is at or above 80% of the free limit.")
    if _pct(storage_gb, FREE_STORAGE_GB) >= WARNING_PCT:
        warnings.append(f"Storage {storage_gb}/{FREE_STORAGE_GB} GB is at or above 80% of the free limit.")
    return {
        "region": region, "prefix": prefix,
        "free_limits": {"rcu": FREE_RCU, "wcu": FREE_WCU, "storage_gb": FREE_STORAGE_GB},
        "totals": {"provisioned_rcu": total_rcu, "provisioned_wcu": total_wcu,
                   "size_gb": storage_gb, "item_count": total_items,
                   "rcu_pct_of_free": _pct(total_rcu, FREE_RCU),
                   "wcu_pct_of_free": _pct(total_wcu, FREE_WCU),
                   "storage_pct_of_free": _pct(storage_gb, FREE_STORAGE_GB)},
        "tables": tables, "consumed_24h": consumed,
        "warnings": warnings,
        "note": "Approximate: DynamoDB does not report free-tier usage directly. "
                "Provisioned capacity and storage are from DescribeTable; "
                "consumed units are a 24-hour CloudWatch sum when available. "
                "AWS Billing is the source of truth for charges.",
    }


def is_admin(user_id: str | None) -> bool:
    """Admin is the Google sub set in ADMIN_GOOGLE_SUB, or the full
    google#sub in ADMIN_USER_ID. If neither is set, nobody is admin
    (the usage/search-sync endpoints return 503, not open access)."""
    if not user_id:
        return False
    admin_sub = os.environ.get("ADMIN_GOOGLE_SUB", "").strip()
    admin_user = os.environ.get("ADMIN_USER_ID", "").strip()
    if admin_sub and user_id == f"google#{admin_sub}":
        return True
    if admin_user and user_id == admin_user:
        return True
    return False
