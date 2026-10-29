# Shibu AI — AWS setup (Steps 2 and 3, DynamoDB only)

This file is local only. Nothing here touches GitHub.
GitHub is touched only when Kartik says "merge".

## 1. Create the AWS account (once, ~10 minutes)

1. Go to aws.amazon.com and click "Create an AWS Account".
2. Use an email you control, set a strong root password, and pick
   "Personal" account type. AWS asks for a card for verification —
   the DynamoDB free allowance itself is $0, the card is for identity
   and for anything you later choose to pay for.
3. Verify the email and phone code AWS sends you.
4. Sign in as root, turn on MFA for the root user:
   top-right account name > Security credentials > MFA > Assign MFA device.
   Do this before anything else. Do not use the root user day to day.

## 2. Region choice (decide BEFORE running the script)

Default in the script: `ap-south-1` (Mumbai), closest to Patna and
best for testing from your PC.

Note from the design: Shibu's live server is on Render in Oregon,
so in production a US-West region would be lower-latency between
Render and DynamoDB. Changing region later means creating new tables
and moving data, so if you want US-West for production, change
`--region` on the very first run. Tables are not created yet, so
changing it now costs nothing.

## 3. Create an IAM user to run the setup (do NOT paste keys in chat)

1. AWS Console > IAM > Users > Create user. Name: `shibu-setup`.
2. Do NOT tick "console access" for this user — it only needs keys.
3. Attach policy: for this one-time setup, `AmazonDynamoDBFullAccess`
   is the simplest correct choice (it can create tables, TTL and PITR).
   This is a setup-only user, not the app user.
4. Create access key > "Command Line Interface (CLI)" > Create.
   Copy the Access Key ID and Secret Access Key into a safe place.
   The secret is shown once. Never paste either key in chat, in
   GitHub, or in the frontend.
5. Step 6 (later, not now) will replace this with a least-privilege
   app user that can only touch Shibu's 4 tables. Do not reuse the
   setup user as the app user long-term.

## 4. Run the table script (one command, idempotent)

On your Windows PC, in the project folder (the inner folder that
contains `app.py`):

```powershell
pip install -r infra/requirements.txt
$env:AWS_ACCESS_KEY_ID="paste-your-setup-key-id-here"
$env:AWS_SECRET_ACCESS_KEY="paste-your-setup-secret-here"
python infra/create_dynamodb_tables.py --region ap-south-1
```

What it creates (provisioned, 21 RCU / 21 WCU total, under 25/25 free):

- `shibu-users` — 2/2, PK `user_id`
- `shibu-conversations` — 3/3, PK `user_id`, SK `conversation_id`,
  GSI `by-updated` 3/3
- `shibu-messages` — 8/8, PK `conversation_id`, SK `created_at_id`,
  GSI `learning-index` 2/2 (sparse, learning only)
- `shibu-cached-answers` — 3/3, PK `query_hash`, TTL on `expires_at`

Encryption at rest uses the AWS-owned key ($0). PITR is on by
default (~$0.20/GB-month, cents at this size); add `--no-pitr` to
stay strictly free.

To preview without calling AWS:

```powershell
python infra/create_dynamodb_tables.py --region ap-south-1 --dry-run
```

To run it again safely (for example after a failed first run):

```powershell
python infra/create_dynamodb_tables.py --region ap-south-1
```

Existing tables are left alone; it does not delete or recreate them.

## 5. Verify in the console (2 minutes)

AWS Console > DynamoDB > Tables, region ap-south-1 (top-right must
match the region you used). You should see all 4 tables, Status
Active. Click each one and check:

- `shibu-conversations` > Indexes: `by-updated` Active
- `shibu-messages` > Indexes: `learning-index` Active
- `shibu-cached-answers` > Additional settings > Time to live:
  Enabled, attribute `expires_at`
- Any table > Additional settings > Point-in-time recovery: On
  (unless you used `--no-pitr`)
- Any table > Additional settings > Encryption: AWS owned key

If a table is still "Creating", wait a minute and refresh — do not
run the script a second time to "fix" it; the waiter already waits
for Active, and re-running is safe anyway.

## 6. Step 6: least-privilege app user (replace shibu-setup)

After the tables are Active and the app key is confirmed working,
the setup user must not be the app's day-to-day key.

1. In the same PowerShell window, with the SETUP key set:
   ```powershell
   python infra/create_app_iam_user.py --region ap-south-1
   ```
   This creates user `shibu-app` (no console access) and attaches
   only the inline policy in `infra/shibu_app_policy.json`:
   Get/Put/Update/Delete/Query/Batch/Describe on exactly the 4
   shibu tables and their 2 GSIs, plus read-only CloudWatch
   metrics for the usage page. No CreateTable, no DeleteTable,
   no Scan, no other service, no billing, no IAM changes.
2. When ready to use it, run with `--create-access-key` and copy
   the secret immediately into Render Environment Variables
   (or a local `.env` for testing). The secret is shown once,
   in this terminal only — never in chat, never in git.
3. Test the app with the app key (`DYNAMODB_ENABLED=1`), then
   delete the setup user's access key in IAM > Users >
   `shibu-setup` > Security credentials. Keep the setup user
   itself only until you no longer need to create tables.

## 7. Step 7: cost protection — Budget + billing alarm (user-side)

DynamoDB's free allowance is 25 RCU / 25 WCU / 25 GB in
provisioned mode, and Shibu uses 21/21, so it should stay free —
but a budget is the safety net, not a substitute for checking.

1. Billing > Billing preferences: turn ON "Receive Billing Alerts"
   (root user only; this enables CloudWatch billing alarms).
2. Billing > Budgets > Create budget > Cost budget: $1 per month,
   alert at 80% ($0.80) and at 100% ($1.00), email = your AWS
   account email. This is a notification only — it does not stop
   usage or make a purchase.
3. CloudWatch > Alarms > Create alarm (optional, same threshold):
   metric `AWS/Billing > EstimatedCharges`, currency USD,
   threshold > $0.80 for 1 period, notify the same email.
4. Admin usage page in the app: log in with Google as the admin
   (set `ADMIN_GOOGLE_SUB` to your Google `sub` — your user_id
   in `/api/me` is `google#<sub>`), then open
   `/api/admin/usage`. It shows provisioned RCU/WCU, storage,
   item counts, and a warning at 80% of the free limit, with a
   clear note that the numbers are approximate and AWS Billing
   is the source of truth for charges.

## 8. What you must NOT do yet

- Do not put AWS keys in `.env` for the app yet — Step 4 will add
  the app's own env names (`AWS_REGION`, table names, app IAM keys).
- Do not push these files to GitHub — Kartik said merge only on
  his word. They stay local until he says "merge".
- Do not delete the `shibu-setup` user until Step 6 replaces it.

## 7. Troubleshooting

- `NoCredentialsError` / `InvalidClientTokenId`: the two
  `$env:` lines were set in a different PowerShell window, or a
  key was copied with a space. Set them again in the same window.
- `AccessDenied`: the IAM user does not have DynamoDB creation
  rights — re-check step 3, policy `AmazonDynamoDBFullAccess`.
- Wrong region in console: the region dropdown top-right must match
  `--region`. Tables are per-region; ap-south-1 tables do not show
  in us-east-1.
- `Table already exists`: that is fine, the script skips it.
  If you want a clean restart, delete the table in the console
  first, wait until it is gone, then re-run.
