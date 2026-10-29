"""Step 6: create the least-privilege app IAM user (shibu-app).

Replaces the setup user (shibu-setup, AmazonDynamoDBFullAccess)
for day-to-day app use. The app user can ONLY touch the 4 shibu-
tables and their GSIs, with only Get/Put/Update/Delete/Query/
Batch/Describe — no CreateTable, no DeleteTable, no Scan, no
other AWS service, no billing, no IAM changes.

Usage (PowerShell, in the project folder, with the SETUP key set
in this window only — the setup key is needed once to create
the app user, then it should be deleted or kept offline):
  pip install -r infra/requirements.txt
  $env:AWS_ACCESS_KEY_ID="setup-key-id"
  $env:AWS_SECRET_ACCESS_KEY="setup-secret"
  python infra/create_app_iam_user.py --region ap-south-1

By default this script does NOT create an access key. Add
--create-access-key only when you are ready to copy the secret
immediately into Render Environment Variables (or a local .env
for testing) — the secret is shown ONCE in this terminal, never
written to a file, never printed to chat, never committed.
After the app key works, delete the setup user's access key in
IAM > Users > shibu-setup > Security credentials (or delete the
setup user entirely, once you no longer need to create tables).
Do NOT delete the setup user before the app key is confirmed
working, or you will lock yourself out of creating tables.
"""

import argparse
import json
import os
import sys

POLICY_NAME = "shibu-app-dynamodb"
USER_NAME = "shibu-app"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--create-access-key", action="store_true",
                        help="Also create an access key (secret shown once).")
    args = parser.parse_args()

    import boto3
    from botocore.exceptions import ClientError

    iam = boto3.client("iam", region_name=args.region)
    policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "shibu_app_policy.json")
    with open(policy_path, encoding="utf-8") as f:
        policy_doc = f.read()
    # Validate JSON before sending it to AWS.
    json.loads(policy_doc)

    try:
        iam.create_user(UserName=USER_NAME)
        print(f"Created IAM user: {USER_NAME}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"IAM user already exists: {USER_NAME} (keeping it)")
        else:
            raise

    iam.put_user_policy(UserName=USER_NAME, PolicyName=POLICY_NAME,
                        PolicyDocument=policy_doc)
    print(f"Attached inline policy: {POLICY_NAME}")
    print("This user has NO console access and NO managed full-access policy.")

    if args.create_access_key:
        key = iam.create_access_key(UserName=USER_NAME)["AccessKey"]
        print("\n*** SAVE THIS NOW — IT WILL NOT BE SHOWN AGAIN ***")
        print(f"AWS_ACCESS_KEY_ID={key['AccessKeyId']}")
        print(f"AWS_SECRET_ACCESS_KEY={key['SecretAccessKey']}")
        print("Put these ONLY in Render Environment Variables (or a local .env for testing).")
        print("Never paste them in chat, never commit them, never put them in the frontend.")
    else:
        print("\nNo access key created. To create one later, run with --create-access-key,")
        print("or in IAM > Users > shibu-app > Security credentials > Create access key > CLI.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - user-facing script error
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
