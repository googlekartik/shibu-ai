-- Supabase pgvector search index for Shibu AI (search step).
-- DynamoDB remains the main store and source of truth; this table
-- is rebuildable from the DynamoDB learning-index GSI at any time.
-- Run this in Supabase > SQL Editor, once, in a new project.
-- Free-tier notes: free projects pause after 1 week of inactivity.
-- When Supabase is paused/unreachable, chat keeps working and
-- search degrades to "no past answers found" — it never breaks chat.

create extension if not exists vector;

create table if not exists good_answers (
  message_id text primary key,          -- same ID as the DynamoDB message
  conversation_id text not null,
  user_id text not null,                -- google#<sub> of the answer's owner
  answer_text text not null,            -- capped ~4,000 chars by the indexer
  embedding vector(1536),               -- OpenAI text-embedding-3-small
  -- Keyword fallback, generated from answer_text (never set by the app).
  search_text tsvector generated always as (to_tsvector('english', coalesce(answer_text, ''))) stored,
  rating text,
  has_correction boolean default false,
  approved boolean default true,
  created_at timestamptz default now()
);

create index if not exists good_answers_embedding_idx
  on good_answers using hnsw (embedding vector_cosine_ops);
create index if not exists good_answers_user_idx on good_answers (user_id);
create index if not exists good_answers_conversation_idx on good_answers (conversation_id);
create index if not exists good_answers_search_idx on good_answers using gin (search_text);

-- Row Level Security: the app uses the service_role key server-side
-- (which bypasses RLS), so these policies are a safety net for any
-- other key. Never use the public anon key from the app.
alter table good_answers enable row level security;
drop policy if exists "deny anon" on good_answers;
create policy "deny anon" on good_answers for all to anon using (false);

-- Vector search, user-scoped by default. Admin cross-user search
-- is done by the app with the service_role key, only over approved
-- + consented answers, and is NOT exposed as a database function
-- without a user_id filter.
create or replace function match_good_answers(
  query_embedding vector(1536),
  match_user_id text,
  match_threshold float default 0.7,
  match_count int default 5
)
returns table (
  message_id text,
  conversation_id text,
  user_id text,
  answer_text text,
  similarity float
)
language sql stable
as $$
  select message_id, conversation_id, user_id, answer_text,
         1 - (embedding <=> query_embedding) as similarity
  from good_answers
  where user_id = match_user_id
    and approved = true
    and embedding is not null
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
