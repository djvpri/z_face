-- ============================================================
-- Migration: Tabel catatan untuk orang terdaftar
-- Jalankan di PostgreSQL (psql, DBeaver, atau Railway DB)
-- Pastikan tabel organizations dan users sudah ada
-- ============================================================

-- Pastikan pgvector ada (wajib untuk face embedding)
create extension if not exists vector;

-- Tabel catatan per orang
create table if not exists person_notes (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null references organizations(id) on delete cascade,
  person_name text not null,
  note text,
  file_url text,
  file_name text,
  file_type text,
  created_by uuid references users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Index untuk query cepat per org + person
create index if not exists person_notes_org_person_idx
  on person_notes (org_id, person_name);
