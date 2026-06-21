-- Tautkan wajah terdaftar ke akun login (users), supaya "Login dengan Wajah"
-- bisa benar-benar masuk ke dashboard, bukan cuma token cross-app.
ALTER TABLE faces ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_faces_user_id ON faces(user_id);
