-- ============================================================
-- Migration: Tambah kolom foto ke tabel faces
-- Jalankan di PostgreSQL
-- ============================================================

-- Kolom photo menyimpan thumbnail base64 wajah yang didaftarkan
-- Format: base64 encoded JPEG tanpa prefix data:image
alter table faces add column if not exists photo text;
