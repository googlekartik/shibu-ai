# Developer Panel — Phase 1 setup (analytics + settings)

The developer panel is a **separate Render service** (`shibu-ai-dev`)
running the **same code** with `DEV_PANEL=1`. On it, `/` redirects to
`/dev` and `/api/chat` returns 403. The main service is unchanged
except for three things: it records query/login events, it reads the
three panel settings on every chat request, and it serves the public
announcement banner.

Phase 2 (API-key manager via the Render API) and Phase 3 (multi-model
ensemble) are **not built**. There are no endpoints for them yet.

## 1. What the panel shows

- Overview: total users, logins today / last 7 days, questions today /
  last 7 days, fallback rate, document count, storage backend, RAG
  backend, LLM mode/model.
- Failed questions: questions the AI answered with a fallback, most
  frequent first. This is the daily improvement list — each entry is
  a rule, a document, or a fix waiting to be added.
- Top questions, and users (display name, avatar, message count, last
  active). **No email is stored or shown anywhere**, and user IDs are
  shown shortened to 12 characters.
- Settings (apply immediately, no redeploy): web search on/off,
  ensemble on/off (Phase 3 — stored only, no behaviour change),
  announcement text shown on the main chat.
- "Send to Pihu" box: copies a ready instruction to paste in chat
  with Pihu. The panel cannot send commands to Pihu directly, and no
  panel button can grant Pihu permission to push code — code changes
  still need approval in chat, as usual.

## 2. Owner-only access (fail-closed)

Sign-in is Google OAuth, as on the main site, **plus** an owner
identity check. If none of these is set, the panel is locked for
**everyone**, including logged-in users:

- `DEV_OWNER_EMAIL` — the owner's Google email address, or
- `DEV_OWNER_EMAILS` — comma-separated owner emails, or
- `DEV_OWNER_SUB` — the owner's Google subject ID: either the full
  session value `google#<sub>` or the raw `sub` claim; both forms
  are accepted.

Set exactly one of them in Render Environment **on the dev service
only**. Anonymous requests to `/api/dev/*` get 401; logged-in
non-owners get 403; `/dev` for a non-owner returns 403 plain text.

## 3. Supabase setup (shared persistent database)

Both services must share one database, or the panel would see
nothing: SQLite on Render free is wiped on every redeploy.

1. Create a free Supabase project (region Mumbai if offered, else
   Singapore).
2. Supabase dashboard -> **SQL Editor** -> paste the whole file
   `infra/supabase_schema.sql` -> **Run**. It creates all 9 tables
   (users, documents, document_chunks, conversations, messages,
   user_memories, query_events, login_events, dev_settings) with the
   same columns as the SQLite store, plus indexes, and seeds the
   three default settings. Re-running it is safe.
3. Supabase dashboard -> **Project Settings -> Database** -> copy the
   **pooler** connection string (transaction mode, port 6543). This
   is `DATABASE_URL`.
4. Render -> **each service** (shibu-ai AND shibu-ai-dev) ->
   **Environment** -> add `DATABASE_URL` with that string -> **Save
   Changes**. Never paste it in chat, never commit it, never put it
   in the frontend. The app connects server-side only, via the
   pooler. Never use the Supabase anon key for this app.
5. If `DATABASE_URL` is unset or unreachable, the app silently uses
   SQLite (local file on a PC; ephemeral on Render). Nothing breaks.

Without Supabase, the panel still works on SQLite — but on Render
its numbers reset on every redeploy, and the two services will not
share data. On a PC, SQLite persists and the panel works fully.

## 4. Render setup (second service)

1. Push this code to GitHub (owner approval, as usual).
2. Render dashboard -> the blueprint sync for `render.yaml` will
   propose the new `shibu-ai-dev` service -> approve it. (If Render
   does not prompt, create the service manually: **New -> Web
   Service**, same repo, same build/start commands as the main
   service, and set the env vars below.)
3. On the **dev service only**, set in Environment:
   - `DEV_PANEL=1`
   - `DEV_OWNER_EMAIL=<your Google email>`
   - `DATABASE_URL=<pooler string from step 3 above>`
   - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (same OAuth client
     as the main site), `SECRET_KEY` (any random string; a different
     one from the main service is fine)
   - `GOOGLE_REDIRECT_URI=https://<dev-service-url>/auth/callback`
4. Google Cloud -> project `shibu-ai` -> **APIs & Services ->
   Credentials** -> click the existing OAuth client name (edit, do
   not create a new one) -> **Authorized redirect URIs** -> add
   `https://<dev-service-url>/auth/callback` -> **Save**. The main
   site's redirect URI stays as it is.
5. Wait for the dev service to show **Live / Deployed**.

## 5. Verify

1. Open `https://<dev-service-url>/` while signed out -> it should
   redirect to `/login`.
2. Sign in with the owner Google account -> `/dev` loads with
   overview cards (zeros are normal before any events are recorded).
3. On the main site, ask a question, then reload `/dev` -> the
   question count rises and the question appears under Top questions.
4. In the panel, turn **Web search** off, then ask an unknown
   question on the main site -> the graceful fallback answer is
   returned with no web search. Turn it back on to restore search.
5. In the panel, save an announcement -> it appears as a banner on
   the main chat; dismiss it with × (dismissal lasts for the browser
   session). Clear the announcement in the panel to hide it.

## 6. Settings reference

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `web_search_enabled` | boolean | true | Off: the web fallback never runs (graceful fallback instead); the RAG web-append branch is skipped. Explicit "Search …" requests still search. |
| `ensemble_enabled` | boolean | false | Phase 3 only. Stored and shown; no behaviour change in Phase 1. |
| `announcement` | string, max 500 | "" | Banner text on the main chat. Empty hides the banner. |

Only these three keys exist. `POST /api/dev/settings` rejects any
other key (400), a non-boolean for a boolean key (400), and an
announcement over 500 characters (400).

## 7. Retention and privacy

- Query and login events older than 30 days are deleted whenever
  `/api/dev/overview` is loaded (best-effort; a failure is ignored).
- The panel shows aggregated questions and per-user activity for the
  owner only. It is owner-authenticated, never public, and the data
  must not be published or shared. No email addresses are collected
  into these tables at all.
- Deleting a user in future work must also delete their events;
  there is no delete-user endpoint in Phase 1.
