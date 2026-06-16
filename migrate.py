"""
Script migrasi data dari Supabase ke Railway PostgreSQL.
Jalankan dari Railway Shell (service FastAPI):

    export SUPABASE_URL="https://xxx.supabase.co"
    export SUPABASE_KEY="eyJ..."
    python migrate.py

DATABASE_URL sudah otomatis tersedia di Railway Shell.
"""

import json
import os
import sys
from urllib.request import Request, urlopen

import psycopg2

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
NEW_ORG_ID   = "d4c1ff35-6784-4460-9ecf-295e3a262d84"
DATABASE_URL  = os.getenv("DATABASE_URL", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Set env var SUPABASE_URL dan SUPABASE_KEY dulu.")
    sys.exit(1)
if not DATABASE_URL:
    print("ERROR: DATABASE_URL tidak ditemukan.")
    sys.exit(1)


def sb_get(table: str, select: str = "*") -> list:
    url = f"{SUPABASE_URL}/rest/v1/{table}?select={select}&limit=5000"
    req = Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    with urlopen(req) as r:
        return json.loads(r.read())


def vec(emb: list) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


def main():
    print("Menghubungkan ke Railway PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # ---- Faces ----
    print("\n[1/3] Migrasi faces...")
    faces = sb_get("faces", "id,name,title,embedding,created_at")
    print(f"      Ditemukan {len(faces)} entri")
    ok = 0
    for f in faces:
        try:
            cur.execute(
                "INSERT INTO faces (id,name,title,embedding,org_id,created_at) "
                "VALUES (%s::uuid,%s,%s,%s::vector,%s::uuid,%s) "
                "ON CONFLICT (id) DO NOTHING",
                (f["id"], f["name"], f.get("title", ""), vec(f["embedding"]), NEW_ORG_ID, f["created_at"]),
            )
            ok += 1
        except Exception as e:
            print(f"      SKIP {f.get('name')}: {e}")
            conn.rollback()
    conn.commit()
    print(f"      OK — {ok} wajah dipindah")

    # ---- Detection logs ----
    print("\n[2/3] Migrasi detection_logs...")
    try:
        logs = sb_get("detection_logs", "id,name,similarity,det_score,source,created_at,embedding")
        has_emb = True
    except Exception:
        logs = sb_get("detection_logs", "id,name,similarity,det_score,source,created_at")
        has_emb = False
    print(f"      Ditemukan {len(logs)} entri")
    ok = 0
    for l in logs:
        emb = l.get("embedding") if has_emb else None
        try:
            if emb:
                cur.execute(
                    "INSERT INTO detection_logs "
                    "(id,name,similarity,det_score,source,org_id,embedding,created_at) "
                    "VALUES (%s::uuid,%s,%s,%s,%s,%s::uuid,%s::vector,%s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (l["id"], l.get("name"), l.get("similarity"), l.get("det_score"),
                     l.get("source"), NEW_ORG_ID, vec(emb), l["created_at"]),
                )
            else:
                cur.execute(
                    "INSERT INTO detection_logs "
                    "(id,name,similarity,det_score,source,org_id,created_at) "
                    "VALUES (%s::uuid,%s,%s,%s,%s,%s::uuid,%s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (l["id"], l.get("name"), l.get("similarity"), l.get("det_score"),
                     l.get("source"), NEW_ORG_ID, l["created_at"]),
                )
            ok += 1
        except Exception as e:
            print(f"      SKIP log {l.get('id')}: {e}")
            conn.rollback()
    conn.commit()
    print(f"      OK — {ok} riwayat dipindah")

    # ---- App settings ----
    print("\n[3/3] Migrasi app_settings...")
    try:
        settings = sb_get("app_settings", "key,value")
        for s in settings:
            cur.execute(
                "INSERT INTO app_settings (org_id,key,value) VALUES (%s::uuid,%s,%s) "
                "ON CONFLICT (org_id,key) DO UPDATE SET value=EXCLUDED.value",
                (NEW_ORG_ID, s["key"], s["value"]),
            )
        conn.commit()
        print(f"      OK — {len(settings)} pengaturan dipindah")
    except Exception as e:
        print(f"      Dilewati: {e}")

    cur.close()
    conn.close()
    print("\nMigrasi selesai!")


if __name__ == "__main__":
    main()
