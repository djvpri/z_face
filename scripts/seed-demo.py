# Seed data DEMO untuk ZFace — mengisi organisasi akun demo dengan wajah
# terdaftar + riwayat deteksi (+ beberapa tanda tangan) untuk tampilan demo.
#
# PENTING: embedding DIBUAT ACAK (bukan wajah asli), jadi daftar & statistik
# terisi untuk demo visual, TAPI pengenalan wajah live tidak akan cocok.
#
# Pakai psycopg2 + DATABASE_URL (Railway/pgvector). IDEMPOTENT / RESET MANUAL:
# tiap dijalankan, faces/detection_logs/signatures org ini DIHAPUS lalu diisi
# ulang (organisasi & user TIDAK dihapus). Reset:
#   python scripts/seed-demo.py
# Target org: env DEMO_EMAIL (member) -> org; fallback org bernama "demo";
# fallback org pertama.

import os
import random
import datetime
try:
    import psycopg2
except ImportError:
    import psycopg as psycopg2  # fallback

DEMO_EMAIL = os.environ.get("DEMO_EMAIL", "demo@zomet.my.id")
DB = os.environ.get("DATABASE_URL")
if not DB:
    raise SystemExit("DATABASE_URL tidak diset")

NAMA = ["Budi Santoso", "Sari Dewi", "Andi Pratama", "Rina Wijaya", "Eko Nugroho", "Maya Putri",
        "Hendra Saputra", "Lia Anggraini", "Dimas Permana", "Nadia Sari", "Rizky Hidayat", "Wulan Maharani",
        "Bayu Setiawan", "Citra Lestari"]
TITLES = ["", "", "Nasabah Prioritas", "Karyawan", "Direktur", "Nasabah"]
SOURCES = ["realtime", "realtime", "upload"]


def vec(n):
    return "[" + ",".join(f"{random.gauss(0, 0.05):.5f}" for _ in range(n)) + "]"


def main():
    conn = psycopg2.connect(DB)
    conn.autocommit = False
    cur = conn.cursor()

    # 1. Org target
    org_id = None
    cur.execute(
        "SELECT o.id FROM organizations o JOIN org_members m ON m.org_id=o.id "
        "JOIN users u ON u.id=m.user_id WHERE lower(u.email)=lower(%s) LIMIT 1",
        (DEMO_EMAIL,),
    )
    r = cur.fetchone()
    if r:
        org_id = r[0]
    if not org_id:
        cur.execute("SELECT id, name FROM organizations WHERE name ILIKE '%demo%' ORDER BY created_at LIMIT 1")
        r = cur.fetchone()
        if r:
            org_id = r[0]
    if not org_id:
        cur.execute("SELECT id, name FROM organizations ORDER BY created_at LIMIT 1")
        r = cur.fetchone()
        if r:
            org_id = r[0]
    if not org_id:
        raise SystemExit("Tidak ada organisasi di ZFace. Buat organisasi dulu.")
    cur.execute("SELECT name FROM organizations WHERE id=%s", (org_id,))
    print(f"Target org: {cur.fetchone()[0]} [{org_id}]")

    # 2. RESET
    cur.execute("DELETE FROM detection_logs WHERE org_id=%s", (org_id,))
    cur.execute("DELETE FROM faces WHERE org_id=%s", (org_id,))
    cur.execute("DELETE FROM signatures WHERE org_id=%s", (org_id,))
    print("Data demo lama dibersihkan.")

    # 3. Faces
    faces = random.sample(NAMA, 12)
    for nm in faces:
        cur.execute(
            "INSERT INTO faces (name, title, greet_exempt, embedding, org_id) VALUES (%s,%s,%s,%s::vector,%s)",
            (nm, random.choice(TITLES), random.random() < 0.1, vec(512), org_id),
        )
    print(f"Faces: {len(faces)}")

    # 4. Detection logs (~40, ~30 hari)
    now = datetime.datetime.now()
    logs = 0
    for _ in range(40):
        recognized = random.random() < 0.8
        nm = random.choice(faces) if recognized else "Unknown"
        sim = round(random.uniform(0.62, 0.95), 3) if recognized else round(random.uniform(0.2, 0.45), 3)
        created = now - datetime.timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        cur.execute(
            "INSERT INTO detection_logs (name, similarity, det_score, source, embedding, org_id, created_at) "
            "VALUES (%s,%s,%s,%s,%s::vector,%s,%s)",
            (nm, sim, round(random.uniform(0.85, 0.99), 3), random.choice(SOURCES), vec(512), org_id, created),
        )
        logs += 1
    print(f"Detection logs: {logs}")

    # 5. Signatures (5) — opsional
    for nm in random.sample(NAMA, 5):
        cur.execute(
            "INSERT INTO signatures (name, embedding, org_id) VALUES (%s,%s::vector,%s)",
            (nm, vec(3780), org_id),
        )

    conn.commit()
    print("✅ Seed demo ZFace selesai (catatan: embedding acak, pengenalan live tidak cocok).")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
