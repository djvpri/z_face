-- ============================================================
-- Setup database Face ID untuk Supabase
-- Jalankan seluruh file ini di: Supabase Dashboard > SQL Editor
-- ============================================================

-- 1. Aktifkan ekstensi pgvector
create extension if not exists vector;

-- 2. Tabel penyimpanan embedding wajah
--    Embedding InsightFace buffalo_l berukuran 512 dimensi.
--    Satu orang boleh punya beberapa baris (beberapa foto) — ini
--    justru meningkatkan akurasi identifikasi.
create table if not exists faces (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  embedding vector(512) not null,
  created_at timestamptz not null default now()
);

-- 3. Fungsi pencarian kemiripan (cosine similarity)
--    Mengembalikan kandidat paling mirip di atas threshold.
create or replace function match_faces(
  query_embedding vector(512),
  match_threshold float,
  match_count int
)
returns table (id uuid, name text, similarity float)
language sql stable
as $$
  select
    faces.id,
    faces.name,
    1 - (faces.embedding <=> query_embedding) as similarity
  from faces
  where 1 - (faces.embedding <=> query_embedding) > match_threshold
  order by faces.embedding <=> query_embedding
  limit match_count;
$$;

-- 4. Index HNSW agar pencarian tetap cepat saat data membesar
create index if not exists faces_embedding_idx
  on faces using hnsw (embedding vector_cosine_ops);

-- 5. (Opsional tapi disarankan) Aktifkan RLS.
--    Backend memakai service key sehingga tetap bisa mengakses penuh,
--    sementara akses publik via anon key tertutup.
alter table faces enable row level security;
