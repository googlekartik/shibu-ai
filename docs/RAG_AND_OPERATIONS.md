# Shibu AI cloud RAG operations

## What the production path does

1. A signed-in user uploads a PDF, DOCX, XLSX, TXT, Markdown, HTML or CSV file.
2. The existing extractor reads and normalises it; a preview remains available
   even if semantic indexing is temporarily unavailable.
3. RAG chunks the text, records provenance (source hash + embedding version),
   creates hosted embeddings for dense Pinecone indexes, and writes to the
   user's Pinecone namespace.
4. A revised file with the same name replaces the prior document/vector set.
   An identical repeat upload is skipped without making embedding calls.
5. Retrieval is restricted to that user's namespace. Optional exact metadata
   filters can restrict a search to `document_id` or `source_name`.
6. The response includes source name, chunk number, relevance score and
   citation IDs. The model is instructed to treat retrieved text as untrusted
   reference data, not commands.

`RAG_BACKEND=auto` uses Pinecone only when both Pinecone variables are set;
otherwise it uses the clearly labelled SQLite token-overlap fallback. SQLite
is not represented as semantic search.

## Render environment variables

Set these in the Render dashboard, never in Git or browser JavaScript:

| Variable | Required when | Purpose |
|---|---|---|
| `SECRET_KEY` | Always | Flask session signing; use a long random value. |
| `OPENAI_API_KEY` | OpenAI chat or OpenAI embeddings | Hosted GPT-4o-mini and/or embeddings. |
| `HF_TOKEN` | Hugging Face fallback | Hosted HF Inference Providers token. |
| `PINECONE_API_KEY` | Pinecone RAG | Existing Pinecone account authentication. |
| `PINECONE_INDEX_NAME` | Pinecone RAG | Existing index only; this app never creates an index. |
| `PINECONE_INDEX_TYPE` | Pinecone RAG | `dense` or `integrated`, exactly matching the existing index. |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | Dense index | Must match its vector dimension. |
| `CLOUD_ONLY=true` | Render | Prevents local model code paths. |

Use `GET /api/rag/status` after deployment. It validates the configured
Pinecone index dimension and metric without writing to it. If it reports a
dimension mismatch, stop: create no resources automatically. Either select a
compatible existing embedding model or approve a new index and a complete
re-indexing migration.

## Indexing and re-indexing

- `POST /api/rag/ingest`: file multipart or JSON text/source name.
- `POST /api/rag/search`: accepts `q`, `top_k` (1–20), and optional
  `filters.document_id` or `filters.source_name`.
- `POST /api/rag/documents/<id>/reindex`: removes only that user's existing
  vectors and writes its stored document content again.
- `DELETE /api/rag/documents/<id>`: removes only that user's vector metadata
  and local document record.
- `GET /api/rag/metrics`: process-local aggregate latency/error/count metrics;
  it never exposes document text or user questions.

For an embedding-model change, set a new `EMBEDDING_VERSION`, confirm index
compatibility through `/api/rag/status`, then re-index each document. A new
embedding dimension requires a separately approved index—it cannot be changed
in place.

## Cost controls

- The default OpenAI embedding model is `text-embedding-3-small`; use the
  provider's current pricing page before enabling it.
- Repeated same-content uploads are deduplicated; query embeddings have a short
  process-local TTL cache; upserts and embedding requests are batched.
- `CLOUD_RETRY_ATTEMPTS` is bounded to 1–5 and retries only transient errors.
- Client-side dense + lexical hybrid scoring has no extra provider call.
- Hosted reranking is opt-in (`RAG_RERANK_ENABLED=false` by default) because
  it may have additional usage cost.
- `CLOUD_REQUEST_LIMIT` is a per-process safety limit. Add a Render/edge rate
  limit for multi-instance production deployments.
- The application never creates Pinecone indexes, cloud GPU jobs, or paid
  training resources.

## Persistence, backup and rollback

Pinecone vectors survive a Render redeploy, but the SQLite metadata/full-text
database and uploaded files do not on an ephemeral Render filesystem. For
durable document re-indexing, attach a Render persistent disk or move the
metadata/document store to an approved managed service before relying on it in
production. This project does not provision either one.

Before an embedding migration, retain a Pinecone backup/export using the
Pinecone account controls, record the current index name, embedding version and
environment settings, and test a small document sample. Roll back by restoring
the previous environment variables/model version and pointing the app back at
the unchanged prior index. Do not delete the old index until retrieval quality
and citations have been accepted.
