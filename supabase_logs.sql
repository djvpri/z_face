-- ============================================================
-- Riwayat deteksi wajah (dikenal & tidak dikenal)
-- Jalankan file ini di: Supabase Dashboard > SQL Editor
-- ============================================================

create table if not exists detection_logs (
  id uuid primary key default gen_random_uuid(),
  name text,
  similarity float,
  det_score float,
  source text not null,
  created_at timestamptz not null default now()
);

create index if not exists detection_logs_created_at_idx
  on detection_logs (created_at desc);

alter table detection_logs enable row level security;
