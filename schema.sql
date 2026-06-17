-- ZFace schema untuk Railway PostgreSQL
-- Jalankan sekali di Railway PostgreSQL console

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS organizations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    plan        TEXT DEFAULT 'starter',
    quota_faces INTEGER DEFAULT 500,
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS org_members (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT DEFAULT 'member',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(org_id, user_id)
);

CREATE TABLE IF NOT EXISTS faces (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT NOT NULL,
    title        TEXT DEFAULT '',
    greet_exempt BOOLEAN NOT NULL DEFAULT FALSE,
    embedding    vector(512),
    org_id       UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS detection_logs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT,
    similarity FLOAT,
    det_score  FLOAT,
    source     TEXT,
    org_id     UUID REFERENCES organizations(id) ON DELETE CASCADE,
    embedding  vector(512),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app_settings (
    id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    UNIQUE(org_id, key)
);

CREATE TABLE IF NOT EXISTS signatures (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    embedding  vector(3780),
    org_id     UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION match_faces(
    query_embedding vector(512),
    match_threshold float,
    match_count     int,
    filter_org_id   uuid
)
RETURNS TABLE (id uuid, name text, title text, similarity float)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.name,
        f.title,
        (1 - (f.embedding <=> query_embedding))::float AS similarity
    FROM faces f
    WHERE f.org_id = filter_org_id
      AND (1 - (f.embedding <=> query_embedding)) > match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION match_signatures(
    query_embedding vector(3780),
    match_threshold float,
    match_count     int,
    filter_org_id   uuid
)
RETURNS TABLE (id uuid, name text, similarity float)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id,
        s.name,
        (1 - (s.embedding <=> query_embedding))::float AS similarity
    FROM signatures s
    WHERE s.org_id = filter_org_id
      AND (1 - (s.embedding <=> query_embedding)) > match_threshold
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
