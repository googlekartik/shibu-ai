# Render deployment checklist

1. Push this project to a Git repository and create/update a Render Python web
   service from that repository.
2. Render uses `pip install -r requirements.txt` and the production Gunicorn
   start command in `render.yaml`. It binds to Render's `PORT` and exposes
   `/api/health` as the health check.
3. Add `SECRET_KEY`, then the hosted provider secrets you actually use. Add
   Pinecone values only when the existing index has been confirmed. Do not put
   secrets in `render.yaml`, `.env.example`, frontend files or commit history.
4. Keep `CLOUD_ONLY=true`. No CUDA, local PyTorch model or local model download
   is part of the production dependency set.
5. Open `/api/health`, `/api/llm/status`, and `/api/rag/status`. The last must
   say `live_pinecone: ok` before a production ingest test; a mismatch is a
   configuration blocker, not a condition to auto-fix.
6. Upload one non-sensitive test document while signed in, ask a fact found in
   it, verify its displayed source citation, update that file, and verify the
   old fact no longer retrieves.

Gunicorn uses one worker because the bundled SQLite fallback is file based. If
you scale beyond one worker/instance, first move document metadata and job
coordination to a managed shared datastore and use a durable queue for large
ingestion jobs. No background worker or paid data service is created by this
project.
