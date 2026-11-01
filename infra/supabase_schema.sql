-- Shibu AI — Supabase (Postgres) schema, Phase 1.
--
-- HOW TO RUN: Supabase dashboard -> SQL Editor -> paste this whole
-- file -> Run. Run it ONCE; every statement is IF NOT EXISTS / ON
-- CONFLICT DO NOTHING, so re-running it is safe.
--
-- CONNECTION: the Flask app connects server-side ONLY, via the
-- DATABASE_URL pooler (transaction mode) connection string, set in
-- Render Environment on BOTH services. Never use the anon key, never
-- put the connection string in code, chat, or the frontend.
--
-- SECURITY (Phase 1): no Row Level Security. The app enforces
-- user scoping itself: every query is filtered by user_id taken
-- from the Google session (google#<sub>), never from a request
-- body. Only the app's server-side role connects; there is no
-- direct browser access to this database.
--
-- Epoch timestamps are DOUBLE PRECISION, matching the SQLite REAL
-- columns, so the same code and the same values work on both
-- backends. No email column exists anywhere: users are keyed by
-- google#<sub> only.

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT DEFAULT '',
    avatar_id TEXT,
    onboarding_completed INTEGER DEFAULT 0,
    created_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    source_name TEXT NOT NULL,
    source_type TEXT DEFAULT '',
    status TEXT DEFAULT 'indexed',
    chunk_count INTEGER DEFAULT 0,
    error TEXT DEFAULT '',
    full_text TEXT DEFAULT '',
    source_key TEXT DEFAULT '',
    content_sha256 TEXT DEFAULT '',
    embedding_provider TEXT DEFAULT '',
    embedding_model TEXT DEFAULT '',
    embedding_version TEXT DEFAULT '',
    created_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_source ON documents(user_id, source_key);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    user_id TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    text TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON document_chunks(user_id, document_id);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    title TEXT DEFAULT '',
    created_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    intent TEXT DEFAULT '',
    rule_id TEXT DEFAULT '',
    sources_json TEXT DEFAULT '[]',
    created_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS user_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    fact_text TEXT NOT NULL,
    created_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id, created_at);

-- Developer panel (Phase 1): analytics events + settings.
CREATE TABLE IF NOT EXISTS query_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question TEXT DEFAULT '',
    intent TEXT DEFAULT '',
    rule_id TEXT DEFAULT '',
    confidence DOUBLE PRECISION DEFAULT 0,
    is_fallback INTEGER DEFAULT 0,
    processing_ms DOUBLE PRECISION DEFAULT 0,
    created_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_query_events_created ON query_events(created_at);

CREATE TABLE IF NOT EXISTS login_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_login_events_created ON login_events(created_at);

CREATE TABLE IF NOT EXISTS dev_settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT DEFAULT '',
    updated_at DOUBLE PRECISION
);

-- Seed the three allowlisted settings with their defaults.
-- ON CONFLICT DO NOTHING: an owner's saved values are never reset.
INSERT INTO dev_settings (setting_key, setting_value, updated_at) VALUES
    ('web_search_enabled', '1', EXTRACT(EPOCH FROM now())),
    ('ensemble_enabled', '0', EXTRACT(EPOCH FROM now())),
    ('announcement', '', EXTRACT(EPOCH FROM now()))
ON CONFLICT (setting_key) DO NOTHING;
