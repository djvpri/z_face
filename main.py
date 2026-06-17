"""
ZFace API — Identifikasi orang dari foto
Stack: FastAPI + InsightFace (buffalo_l) + Railway PostgreSQL + pgvector
"""

import datetime
import json
import os
import time

import bcrypt
import cv2
import jwt
import numpy as np
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
# Domain yang boleh akses API (pisahkan dengan koma). Kosong = hanya same-origin
# (app web disajikan dari server yang sama, jadi tidak butuh CORS lintas-domain).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# ----- Paket langganan -----
# quota_faces: batas jumlah wajah (0 = tak terbatas).
# features.signature: izin pakai identifikasi tanda tangan.
PLANS = {
    "starter":    {"quota_faces": 500,  "features": {"signature": False}},
    "pro":        {"quota_faces": 2000, "features": {"signature": True}},
    "enterprise": {"quota_faces": 0,    "features": {"signature": True}},
}
PLAN_DAYS = 30   # masa aktif default saat buat/perpanjang org
GRACE_DAYS = 7   # masa tenggang setelah expired sebelum dikunci total


def plan_quota(plan: str) -> int:
    return PLANS.get(plan, PLANS["starter"])["quota_faces"]


def plan_features(plan: str) -> dict:
    return PLANS.get(plan, PLANS["starter"])["features"]

db_pool: psycopg2.pool.ThreadedConnectionPool | None = None
if DATABASE_URL:
    db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DATABASE_URL)

# ----- Model InsightFace -----
print("Memuat model InsightFace (buffalo_l)...")
from insightface.app import FaceAnalysis  # noqa: E402

face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_engine.prepare(ctx_id=0, det_size=(640, 640))
print("Model siap.")

app = FastAPI(title="ZFace API", version="1.0.0")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    print("CORS: lintas-domain dimatikan (hanya same-origin). Set ALLOWED_ORIGINS bila perlu.")


# ----- Rate limiting login (anti brute-force) -----
LOGIN_MAX_FAILS = 5          # maksimal gagal sebelum diblokir
LOGIN_WINDOW = 15 * 60       # jendela waktu (detik)
_login_fails: dict[str, list[float]] = {}


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def login_rate_check(ip: str):
    now = time.time()
    fails = [t for t in _login_fails.get(ip, []) if now - t < LOGIN_WINDOW]
    _login_fails[ip] = fails
    if len(fails) >= LOGIN_MAX_FAILS:
        wait = int((LOGIN_WINDOW - (now - fails[0])) / 60) + 1
        raise HTTPException(429, f"Terlalu banyak percobaan login. Coba lagi dalam {wait} menit.")


def login_record_fail(ip: str):
    _login_fails.setdefault(ip, []).append(time.time())


def login_record_success(ip: str):
    _login_fails.pop(ip, None)


# ----- DB helpers -----

def require_db():
    if db_pool is None:
        raise HTTPException(500, "Database belum dikonfigurasi (DATABASE_URL)")


def db_one(sql: str, params=()):
    """Jalankan query dan kembalikan satu baris sebagai dict, atau None."""
    require_db()
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


def db_all(sql: str, params=()):
    """Jalankan query dan kembalikan semua baris sebagai list of dict."""
    require_db()
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


def db_run(sql: str, params=()):
    """Jalankan INSERT/UPDATE/DELETE dan kembalikan baris RETURNING."""
    require_db()
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


def vec(embedding: list[float]) -> str:
    """Konversi embedding list ke format string pgvector."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def log_audit(actor: str, action: str, detail: str = "", org_id: str | None = None):
    """Catat aksi penting (buat/hapus/ubah) ke audit_log. Tidak menggagalkan
    aksi utama bila pencatatan gagal."""
    try:
        db_run(
            "INSERT INTO audit_log (actor, action, detail, org_id) VALUES (%s, %s, %s, %s)",
            (actor, action, detail, org_id),
        )
    except Exception:
        pass


# ----- Auth helpers -----

def get_session(authorization: str = Header(default="")):
    """Verifikasi JWT dan kembalikan {user_id, org_id, role}."""
    if not JWT_SECRET:
        raise HTTPException(503, "Auth belum dikonfigurasi (JWT_SECRET)")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token tidak ditemukan. Silakan login.")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload["sub"]
    except Exception:
        raise HTTPException(401, "Sesi tidak valid atau sudah berakhir. Silakan login kembali.")
    row = db_one(
        "SELECT m.org_id, m.role, o.plan, o.active, o.expires_at "
        "FROM org_members m JOIN organizations o ON o.id = m.org_id "
        "WHERE m.user_id = %s LIMIT 1",
        (user_id,),
    )
    if not row:
        raise HTTPException(403, "Akun tidak terdaftar di organisasi manapun. Hubungi admin.")
    if row.get("active") is False:
        raise HTTPException(403, "Organisasi nonaktif. Hubungi admin.")

    plan = row.get("plan") or "starter"
    expires_at = row.get("expires_at")
    sub_status = "active"
    if expires_at is not None:
        now = datetime.datetime.now(datetime.timezone.utc)
        if now > expires_at:
            if now > expires_at + datetime.timedelta(days=GRACE_DAYS):
                raise HTTPException(402, "Langganan organisasi sudah berakhir. Hubungi admin untuk perpanjangan.")
            sub_status = "grace"

    return {
        "user_id": user_id,
        "org_id": str(row["org_id"]),
        "role": row["role"],
        "plan": plan,
        "sub_status": sub_status,
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


def require_admin(x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Akses ditolak")


# Aksi kelola data (edit/hapus/ubah pengaturan) hanya untuk owner & admin.
MANAGE_ROLES = ("owner", "admin")


def require_role(session: dict, allowed: tuple):
    if session.get("role") not in allowed:
        raise HTTPException(403, "Role Anda tidak punya izin untuk aksi ini")


def require_feature(session: dict, feature: str):
    if not plan_features(session.get("plan", "starter")).get(feature, False):
        raise HTTPException(403, f"Fitur ini tidak tersedia di paket {session.get('plan', '')}. Upgrade paket untuk memakainya.")


# ----- Image helpers -----

async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, "File kosong")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "File bukan gambar yang valid (gunakan JPG/PNG)")
    max_side = 1600
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_faces(img: np.ndarray):
    return face_engine.get(img)


def largest_face(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)
    return max(faces, key=area)


# ----- Signature (tanda tangan) helpers -----
# Tanda tangan tidak punya model embedding seperti wajah, jadi dipakai
# pendekatan CV klasik: binarisasi Otsu -> potong ke area tinta ->
# normalisasi ukuran ke kanvas tetap -> deskriptor HOG -> L2-normalize.
# Hasilnya vektor 3780-dim yang disimpan di pgvector, sehingga
# pencocokan memakai cosine similarity yang sama seperti wajah.
SIG_W, SIG_H = 256, 128
SIG_DIM = 3780
_hog = cv2.HOGDescriptor((SIG_W, SIG_H), (32, 32), (16, 16), (16, 16), 9)


def preprocess_signature(img: np.ndarray):
    """Ubah foto tanda tangan jadi kanvas biner ternormalisasi 256x128."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu, tinta gelap -> dibalik jadi tinta putih di latar hitam
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(th)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = th[y:y + h, x:x + w]
    scale = min(SIG_W / w, SIG_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIG_H, SIG_W), dtype=np.uint8)
    ox, oy = (SIG_W - nw) // 2, (SIG_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return cv2.GaussianBlur(canvas, (3, 3), 0)


def signature_embedding(img: np.ndarray):
    """Kembalikan embedding tanda tangan (list 3780 float, L2-normalized) atau None."""
    canvas = preprocess_signature(img)
    if canvas is None:
        return None
    desc = _hog.compute(canvas).flatten().astype(float)
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm
    return desc.tolist()


# ----- Auth endpoints (public) -----

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request):
    if not JWT_SECRET:
        raise HTTPException(503, "Auth belum dikonfigurasi (JWT_SECRET)")
    ip = client_ip(request)
    login_rate_check(ip)
    row = db_one("SELECT id, password_hash FROM users WHERE email = %s", (body.email,))
    if not row:
        login_record_fail(ip)
        raise HTTPException(401, "Email atau password salah")
    pw_ok = bcrypt.checkpw(body.password.encode(), row["password_hash"].encode())
    if not pw_ok:
        login_record_fail(ip)
        raise HTTPException(401, "Email atau password salah")
    login_record_success(ip)
    user_id = str(row["id"])
    token = jwt.encode(
        {
            "sub": user_id,
            "email": body.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"access_token": token, "user": {"id": user_id, "email": body.email}}


@app.get("/api/auth/me")
def get_me(session: dict = Depends(get_session)):
    row = db_one(
        "SELECT name FROM organizations WHERE id = %s::uuid LIMIT 1",
        (session["org_id"],),
    )
    plan = session["plan"]
    return {
        "user_id": session["user_id"],
        "org_id": session["org_id"],
        "role": session["role"],
        "org_name": (row or {}).get("name", ""),
        "plan": plan,
        "quota_faces": plan_quota(plan),
        "features": plan_features(plan),
        "sub_status": session["sub_status"],
        "expires_at": session["expires_at"],
    }


# ----- Admin endpoints -----

class OrgCreateRequest(BaseModel):
    name: str
    plan: str = "starter"


class OrgPlanRequest(BaseModel):
    plan: str


class UserCreateRequest(BaseModel):
    email: str
    password: str
    org_id: str
    role: str = "member"


@app.post("/api/admin/organizations")
def admin_create_org(body: OrgCreateRequest, _=Depends(require_admin)):
    if not body.name.strip():
        raise HTTPException(400, "Nama organisasi tidak boleh kosong")
    if body.plan not in PLANS:
        raise HTTPException(400, "Plan tidak valid")
    # Kuota mengikuti paket; masa aktif otomatis +PLAN_DAYS hari.
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=PLAN_DAYS)
    rows = db_run(
        "INSERT INTO organizations (name, plan, quota_faces, active, expires_at) "
        "VALUES (%s, %s, %s, TRUE, %s) RETURNING *",
        (body.name.strip(), body.plan, plan_quota(body.plan), expires),
    )
    org = rows[0] if rows else None
    if org:
        log_audit("admin", "create_org", f"{org['name']} ({body.plan})", str(org["id"]))
    return {"organization": org}


@app.patch("/api/admin/organizations/{org_id}")
def admin_change_plan(org_id: str, body: OrgPlanRequest, _=Depends(require_admin)):
    if body.plan not in PLANS:
        raise HTTPException(400, "Plan tidak valid")
    rows = db_run(
        "UPDATE organizations SET plan = %s, quota_faces = %s WHERE id = %s::uuid RETURNING *",
        (body.plan, plan_quota(body.plan), org_id),
    )
    if not rows:
        raise HTTPException(404, "Organisasi tidak ditemukan")
    log_audit("admin", "change_plan", f"ke {body.plan}", org_id)
    return {"organization": rows[0]}


@app.post("/api/admin/organizations/{org_id}/renew")
def admin_renew_org(org_id: str, _=Depends(require_admin)):
    """Perpanjang masa aktif +PLAN_DAYS hari (dari tanggal expired bila masih aktif)."""
    org = db_one("SELECT expires_at FROM organizations WHERE id = %s::uuid", (org_id,))
    if org is None:
        raise HTTPException(404, "Organisasi tidak ditemukan")
    now = datetime.datetime.now(datetime.timezone.utc)
    base = org["expires_at"] if org["expires_at"] and org["expires_at"] > now else now
    new_expiry = base + datetime.timedelta(days=PLAN_DAYS)
    rows = db_run(
        "UPDATE organizations SET expires_at = %s, active = TRUE WHERE id = %s::uuid RETURNING *",
        (new_expiry, org_id),
    )
    log_audit("admin", "renew", f"s/d {new_expiry.date().isoformat()}", org_id)
    return {"organization": rows[0]}


@app.get("/api/admin/organizations")
def admin_list_orgs(_=Depends(require_admin)):
    rows = db_all("SELECT * FROM organizations ORDER BY created_at")
    return {"organizations": rows}


@app.get("/api/admin/audit")
def admin_audit(limit: int = 100, _=Depends(require_admin)):
    limit = max(1, min(limit, 500))
    rows = db_all(
        "SELECT actor, action, detail, org_id, created_at FROM audit_log "
        "ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    for r in rows:
        if r.get("org_id") is not None:
            r["org_id"] = str(r["org_id"])
    return {"audit": rows}


@app.post("/api/admin/users")
def admin_create_user(body: UserCreateRequest, _=Depends(require_admin)):
    if body.role not in ("owner", "admin", "member"):
        raise HTTPException(400, "Role tidak valid. Pilih: owner, admin, member")
    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    try:
        rows = db_run(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
            (body.email, pw_hash),
        )
    except Exception as e:
        raise HTTPException(400, f"Gagal membuat user: {e}")
    user_id = str(rows[0]["id"])
    db_run(
        "INSERT INTO org_members (org_id, user_id, role) VALUES (%s, %s, %s)",
        (body.org_id, user_id, body.role),
    )
    log_audit("admin", "create_user", f"{body.email} ({body.role})", body.org_id)
    return {"user_id": user_id, "email": body.email, "org_id": body.org_id, "role": body.role}


# ----- Public endpoints -----

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "buffalo_l",
        "database": "terhubung" if db_pool else "belum dikonfigurasi",
    }


# ----- Protected endpoints -----

@app.post("/api/identify")
async def identify(
    file: UploadFile = File(...),
    threshold: float = Form(0.40),
    session: dict = Depends(get_session),
):
    """Identifikasi semua wajah dalam foto."""
    img = await read_image(file)
    h, w = img.shape[:2]
    faces = detect_faces(img)

    results = []
    for face in faces:
        embedding = face.normed_embedding.astype(float).tolist()
        rows = db_all(
            "SELECT * FROM match_faces(%s::vector, %s, %s, %s::uuid)",
            (vec(embedding), threshold, 3, session["org_id"]),
        )
        matches = [
            {"id": str(m["id"]), "name": m["name"], "similarity": round(float(m["similarity"]), 4)}
            for m in rows
        ]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        results.append({
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "det_score": round(float(face.det_score), 3),
            "matches": matches,
            "best": matches[0] if matches else None,
            "embedding": embedding,
        })

    return {"image": {"width": w, "height": h}, "faces": results}


@app.post("/api/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...),
    session: dict = Depends(get_session),
):
    """Daftarkan wajah baru dari foto."""
    org_id = session["org_id"]
    name = name.strip()
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong")

    org = db_one("SELECT quota_faces, plan FROM organizations WHERE id = %s::uuid", (org_id,)) or {}
    count_row = db_one("SELECT COUNT(*) AS cnt FROM faces WHERE org_id = %s::uuid", (org_id,))
    quota = org.get("quota_faces")
    count = int(count_row["cnt"]) if count_row else 0
    if quota and count >= quota:
        raise HTTPException(400, f"Kuota wajah ({quota}) sudah penuh untuk plan {org.get('plan', '')}")

    img = await read_image(file)
    faces = detect_faces(img)
    if not faces:
        raise HTTPException(400, "Tidak ada wajah terdeteksi di foto")

    face = largest_face(faces)
    embedding = face.normed_embedding.astype(float).tolist()

    rows = db_run(
        "INSERT INTO faces (name, embedding, org_id) VALUES (%s, %s::vector, %s::uuid) RETURNING id",
        (name, vec(embedding), org_id),
    )
    return {
        "id": str(rows[0]["id"]),
        "name": name,
        "faces_detected": len(faces),
        "note": "Lebih dari satu wajah terdeteksi; wajah terbesar yang didaftarkan"
        if len(faces) > 1
        else None,
    }


class EmbeddingRegisterRequest(BaseModel):
    name: str
    embedding: list[float]


@app.post("/api/register-embedding")
def register_embedding(body: EmbeddingRegisterRequest, session: dict = Depends(get_session)):
    """Daftarkan wajah dari embedding yang sudah dihitung sebelumnya."""
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong")
    if len(body.embedding) != 512:
        raise HTTPException(400, "Embedding tidak valid")
    rows = db_run(
        "INSERT INTO faces (name, embedding, org_id) VALUES (%s, %s::vector, %s::uuid) RETURNING id",
        (name, vec(body.embedding), session["org_id"]),
    )
    return {"id": str(rows[0]["id"]), "name": name}


class LogEntryRequest(BaseModel):
    name: str | None = None
    similarity: float | None = None
    det_score: float | None = None
    source: str
    embedding: list[float] | None = None
    photo: str | None = None


@app.post("/api/logs")
def create_log(body: LogEntryRequest, session: dict = Depends(get_session)):
    """Catat satu hasil deteksi ke riwayat."""
    if body.source not in ("identify", "realtime", "guest"):
        raise HTTPException(400, "Source tidak valid")
    has_emb = body.name is None and body.embedding and len(body.embedding) == 512
    if has_emb:
        db_run(
            "INSERT INTO detection_logs (name, similarity, det_score, source, org_id, embedding, photo) "
            "VALUES (%s, %s, %s, %s, %s::uuid, %s::vector, %s)",
            (body.name, body.similarity, body.det_score, body.source, session["org_id"], vec(body.embedding), body.photo),
        )
    else:
        db_run(
            "INSERT INTO detection_logs (name, similarity, det_score, source, org_id, photo) "
            "VALUES (%s, %s, %s, %s, %s::uuid, %s)",
            (body.name, body.similarity, body.det_score, body.source, session["org_id"], body.photo),
        )
    return {"ok": True}


@app.get("/api/logs")
def list_logs(limit: int = 50, session: dict = Depends(get_session)):
    """Riwayat deteksi terbaru, urut dari yang paling baru."""
    limit = max(1, min(limit, 200))
    rows = db_all(
        "SELECT id, name, similarity, det_score, source, created_at, "
        "embedding::text AS embedding, photo "
        "FROM detection_logs WHERE org_id = %s::uuid ORDER BY created_at DESC LIMIT %s",
        (session["org_id"], limit),
    )
    for r in rows:
        r["id"] = str(r["id"])
        if r.get("embedding"):
            try:
                r["embedding"] = json.loads(r["embedding"])
            except Exception:
                r["embedding"] = None
    return {"logs": rows}


DEFAULT_SETTINGS = {
    "greeting_known": "Selamat datang, {nama}!",
    "greeting_unknown": "Halo! Wajah belum terdaftar",
}


class SettingsRequest(BaseModel):
    greeting_known: str | None = None
    greeting_unknown: str | None = None


@app.get("/api/settings")
def get_settings(session: dict = Depends(get_session)):
    """Ambil pengaturan sapaan untuk org ini."""
    rows = db_all(
        "SELECT key, value FROM app_settings WHERE org_id = %s::uuid",
        (session["org_id"],),
    )
    settings = dict(DEFAULT_SETTINGS)
    for row in rows:
        if row["key"] in settings:
            settings[row["key"]] = row["value"]
    return settings


@app.put("/api/settings")
def update_settings(body: SettingsRequest, session: dict = Depends(get_session)):
    """Perbarui pengaturan sapaan."""
    require_role(session, MANAGE_ROLES)
    org_id = session["org_id"]
    updates = body.model_dump(exclude_none=True)
    for key, value in updates.items():
        db_run(
            "INSERT INTO app_settings (org_id, key, value) VALUES (%s::uuid, %s, %s) "
            "ON CONFLICT (org_id, key) DO UPDATE SET value = EXCLUDED.value",
            (org_id, key, value),
        )
    return get_settings(session)


@app.get("/api/people")
def list_people(session: dict = Depends(get_session)):
    """Daftar orang terdaftar, dikelompokkan per nama."""
    rows = db_all(
        "SELECT id, name, title, greet_exempt, created_at FROM faces WHERE org_id = %s::uuid ORDER BY created_at DESC",
        (session["org_id"],),
    )
    grouped: dict[str, dict] = {}
    for row in rows:
        g = grouped.setdefault(row["name"], {
            "name": row["name"],
            "title": row.get("title", ""),
            "greet_exempt": bool(row.get("greet_exempt", False)),
            "photos": 0,
            "entries": [],
        })
        g["photos"] += 1
        g["entries"].append({"id": str(row["id"]), "created_at": row["created_at"]})
    return {"total_entries": len(rows), "people": list(grouped.values())}


class PersonUpdateRequest(BaseModel):
    new_name: str | None = None
    title: str | None = None
    greet_exempt: bool | None = None


@app.patch("/api/people/{name}")
def update_person(name: str, body: PersonUpdateRequest, session: dict = Depends(get_session)):
    """Ganti nama, gelar, dan/atau status pengecualian sapaan satu orang."""
    require_role(session, MANAGE_ROLES)
    sets = []
    params: list = []
    if body.new_name is not None:
        new_name = body.new_name.strip()
        if not new_name:
            raise HTTPException(400, "Nama tidak boleh kosong")
        sets.append("name = %s")
        params.append(new_name)
    if body.title is not None:
        if body.title not in ("", "Bapak", "Ibu"):
            raise HTTPException(400, "Gelar tidak valid")
        sets.append("title = %s")
        params.append(body.title)
    if body.greet_exempt is not None:
        sets.append("greet_exempt = %s")
        params.append(body.greet_exempt)
    if not sets:
        return {"updated_entries": 0}
    params.extend([name, session["org_id"]])
    rows = db_run(
        f"UPDATE faces SET {', '.join(sets)} WHERE name = %s AND org_id = %s::uuid RETURNING id",
        tuple(params),
    )
    return {"updated_entries": len(rows)}


@app.delete("/api/faces/{face_id}")
def delete_face(face_id: str, session: dict = Depends(get_session)):
    require_role(session, MANAGE_ROLES)
    db_run(
        "DELETE FROM faces WHERE id = %s::uuid AND org_id = %s::uuid",
        (face_id, session["org_id"]),
    )
    return {"deleted": face_id}


@app.delete("/api/people/{name}")
def delete_person(name: str, session: dict = Depends(get_session)):
    require_role(session, MANAGE_ROLES)
    rows = db_run(
        "DELETE FROM faces WHERE name = %s AND org_id = %s::uuid RETURNING id",
        (name, session["org_id"]),
    )
    log_audit(session["user_id"], "delete_person", f"{name} ({len(rows)} foto)", session["org_id"])
    return {"deleted_name": name, "deleted_entries": len(rows)}


# ----- Signature (tanda tangan) endpoints -----

@app.post("/api/signatures/identify")
async def identify_signature(
    file: UploadFile = File(...),
    threshold: float = Form(0.55),
    session: dict = Depends(get_session),
):
    """Identifikasi pemilik tanda tangan dari foto/scan."""
    require_feature(session, "signature")
    img = await read_image(file)
    embedding = signature_embedding(img)
    if embedding is None:
        raise HTTPException(400, "Tidak ada goresan tanda tangan terdeteksi di gambar")
    rows = db_all(
        "SELECT * FROM match_signatures(%s::vector, %s, %s, %s::uuid)",
        (vec(embedding), threshold, 3, session["org_id"]),
    )
    matches = [
        {"id": str(m["id"]), "name": m["name"], "similarity": round(float(m["similarity"]), 4)}
        for m in rows
    ]
    return {"matches": matches, "best": matches[0] if matches else None, "embedding": embedding}


@app.post("/api/signatures/register")
async def register_signature(
    name: str = Form(...),
    file: UploadFile = File(...),
    session: dict = Depends(get_session),
):
    """Daftarkan contoh tanda tangan atas nama seseorang."""
    require_feature(session, "signature")
    name = name.strip()
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong")
    img = await read_image(file)
    embedding = signature_embedding(img)
    if embedding is None:
        raise HTTPException(400, "Tidak ada goresan tanda tangan terdeteksi di gambar")
    rows = db_run(
        "INSERT INTO signatures (name, embedding, org_id) VALUES (%s, %s::vector, %s::uuid) RETURNING id",
        (name, vec(embedding), session["org_id"]),
    )
    return {"id": str(rows[0]["id"]), "name": name}


class SignatureEmbeddingRequest(BaseModel):
    name: str
    embedding: list[float]


@app.post("/api/signatures/register-embedding")
def register_signature_embedding(body: SignatureEmbeddingRequest, session: dict = Depends(get_session)):
    """Daftarkan tanda tangan dari embedding yang sudah dihitung (hasil identify)."""
    require_feature(session, "signature")
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong")
    if len(body.embedding) != SIG_DIM:
        raise HTTPException(400, "Embedding tanda tangan tidak valid")
    rows = db_run(
        "INSERT INTO signatures (name, embedding, org_id) VALUES (%s, %s::vector, %s::uuid) RETURNING id",
        (name, vec(body.embedding), session["org_id"]),
    )
    return {"id": str(rows[0]["id"]), "name": name}


@app.get("/api/signatures")
def list_signatures(session: dict = Depends(get_session)):
    """Daftar tanda tangan terdaftar, dikelompokkan per nama."""
    rows = db_all(
        "SELECT id, name, created_at FROM signatures WHERE org_id = %s::uuid ORDER BY created_at DESC",
        (session["org_id"],),
    )
    grouped: dict[str, dict] = {}
    for row in rows:
        g = grouped.setdefault(row["name"], {"name": row["name"], "samples": 0})
        g["samples"] += 1
    return {"total": len(rows), "signatures": list(grouped.values())}


@app.delete("/api/signatures/{name}")
def delete_signature(name: str, session: dict = Depends(get_session)):
    require_role(session, MANAGE_ROLES)
    rows = db_run(
        "DELETE FROM signatures WHERE name = %s AND org_id = %s::uuid RETURNING id",
        (name, session["org_id"]),
    )
    log_audit(session["user_id"], "delete_signature", f"{name} ({len(rows)} contoh)", session["org_id"])
    return {"deleted_name": name, "deleted_samples": len(rows)}


# ----- Static UI -----

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def landing():
    return FileResponse("static/landing.html")


@app.get("/app")
def index():
    return FileResponse("static/index.html")


@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")


@app.get("/sw.js")
def service_worker():
    return FileResponse("static/sw.js", media_type="application/javascript")
