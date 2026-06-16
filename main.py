"""
ZFace API — Identifikasi orang dari foto
Stack: FastAPI + InsightFace (buffalo_l) + Supabase pgvector
Jalankan: uvicorn main:app --reload
"""

import os

import cv2
import jwt
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

supabase = None
supabase_anon = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
if SUPABASE_URL and SUPABASE_ANON_KEY:
    supabase_anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ----- Model InsightFace -----
# Model buffalo_l (~280 MB) otomatis diunduh ke ~/.insightface saat pertama kali dijalankan
print("Memuat model InsightFace (buffalo_l)...")
from insightface.app import FaceAnalysis  # noqa: E402

face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_engine.prepare(ctx_id=0, det_size=(640, 640))
print("Model siap.")

app = FastAPI(title="ZFace API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Helpers -----

def require_db():
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase belum dikonfigurasi. Isi SUPABASE_URL dan SUPABASE_SERVICE_KEY di file .env",
        )


def get_session(authorization: str = Header(default="")):
    """Verifikasi JWT dan kembalikan {user_id, org_id, role}."""
    if not JWT_SECRET:
        raise HTTPException(503, "SUPABASE_JWT_SECRET belum dikonfigurasi di server")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token tidak ditemukan. Silakan login.")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience="authenticated")
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Sesi sudah berakhir, silakan login kembali")
    except Exception:
        raise HTTPException(401, "Token tidak valid")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(401, "Token tidak valid")
    require_db()
    res = supabase.table("org_members").select("org_id, role").eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(403, "Akun tidak terdaftar di organisasi manapun. Hubungi admin.")
    row = res.data[0]
    return {"user_id": user_id, "org_id": row["org_id"], "role": row["role"]}


def require_admin(x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Akses ditolak")


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File kosong")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid (gunakan JPG/PNG)")
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


# ----- Auth endpoints (public) -----

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest):
    if not supabase_anon:
        raise HTTPException(503, "Auth belum dikonfigurasi (SUPABASE_ANON_KEY)")
    try:
        res = supabase_anon.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception:
        raise HTTPException(401, "Email atau password salah")
    return {
        "access_token": res.session.access_token,
        "user": {"id": str(res.user.id), "email": res.user.email},
    }


@app.get("/api/auth/me")
def get_me(session: dict = Depends(get_session)):
    require_db()
    res = supabase.table("organizations").select("name, plan, quota_faces").eq("id", session["org_id"]).execute()
    org = res.data[0] if res.data else {}
    return {
        "user_id": session["user_id"],
        "org_id": session["org_id"],
        "role": session["role"],
        "org_name": org.get("name", ""),
        "plan": org.get("plan", "starter"),
        "quota_faces": org.get("quota_faces", 0),
    }


# ----- Admin endpoints -----

class OrgCreateRequest(BaseModel):
    name: str
    plan: str = "starter"
    quota_faces: int = 500


class UserCreateRequest(BaseModel):
    email: str
    password: str
    org_id: str
    role: str = "member"


@app.post("/api/admin/organizations")
def admin_create_org(body: OrgCreateRequest, _=Depends(require_admin)):
    require_db()
    if not body.name.strip():
        raise HTTPException(400, "Nama organisasi tidak boleh kosong")
    res = supabase.table("organizations").insert({
        "name": body.name.strip(),
        "plan": body.plan,
        "quota_faces": body.quota_faces,
        "active": True,
    }).execute()
    return {"organization": res.data[0] if res.data else None}


@app.get("/api/admin/organizations")
def admin_list_orgs(_=Depends(require_admin)):
    require_db()
    res = supabase.table("organizations").select("*").order("created_at").execute()
    return {"organizations": res.data}


@app.post("/api/admin/users")
def admin_create_user(body: UserCreateRequest, _=Depends(require_admin)):
    require_db()
    if body.role not in ("owner", "admin", "member"):
        raise HTTPException(400, "Role tidak valid. Pilih: owner, admin, member")
    try:
        user_res = supabase.auth.admin.create_user({
            "email": body.email,
            "password": body.password,
            "email_confirm": True,
        })
    except Exception as e:
        raise HTTPException(400, f"Gagal membuat user: {e}")
    user_id = str(user_res.user.id)
    supabase.table("org_members").insert({
        "org_id": body.org_id,
        "user_id": user_id,
        "role": body.role,
    }).execute()
    return {"user_id": user_id, "email": body.email, "org_id": body.org_id, "role": body.role}


# ----- Public endpoints -----

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "buffalo_l",
        "database": "terhubung" if supabase else "belum dikonfigurasi",
    }


# ----- Protected endpoints -----

@app.post("/api/identify")
async def identify(
    file: UploadFile = File(...),
    threshold: float = Form(0.40),
    session: dict = Depends(get_session),
):
    """Identifikasi semua wajah dalam foto."""
    require_db()
    img = await read_image(file)
    h, w = img.shape[:2]
    faces = detect_faces(img)

    results = []
    for face in faces:
        embedding = face.normed_embedding.astype(float).tolist()
        rpc = supabase.rpc(
            "match_faces",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": 3,
                "filter_org_id": session["org_id"],
            },
        ).execute()

        matches = [
            {"id": m["id"], "name": m["name"], "similarity": round(m["similarity"], 4)}
            for m in (rpc.data or [])
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
    """Daftarkan wajah baru. Satu orang boleh didaftarkan beberapa kali
    dengan foto berbeda untuk meningkatkan akurasi."""
    require_db()
    org_id = session["org_id"]
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong")

    org_res = supabase.table("organizations").select("quota_faces, plan").eq("id", org_id).execute()
    org = org_res.data[0] if org_res.data else {}
    count_res = supabase.table("faces").select("id", count="exact").eq("org_id", org_id).execute()
    quota = org.get("quota_faces")
    if quota and count_res.count is not None and count_res.count >= quota:
        raise HTTPException(400, f"Kuota wajah ({quota}) sudah penuh untuk plan {org.get('plan', '')}")

    img = await read_image(file)
    faces = detect_faces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="Tidak ada wajah terdeteksi di foto")

    face = largest_face(faces)
    embedding = face.normed_embedding.astype(float).tolist()

    res = supabase.table("faces").insert({
        "name": name,
        "embedding": embedding,
        "org_id": org_id,
    }).execute()

    return {
        "id": res.data[0]["id"],
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
    require_db()
    org_id = session["org_id"]
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong")
    if len(body.embedding) != 512:
        raise HTTPException(status_code=400, detail="Embedding tidak valid")

    res = supabase.table("faces").insert({
        "name": name,
        "embedding": body.embedding,
        "org_id": org_id,
    }).execute()
    return {"id": res.data[0]["id"], "name": name}


class LogEntryRequest(BaseModel):
    name: str | None = None
    similarity: float | None = None
    det_score: float | None = None
    source: str


@app.post("/api/logs")
def create_log(body: LogEntryRequest, session: dict = Depends(get_session)):
    """Catat satu hasil deteksi ke riwayat."""
    require_db()
    if body.source not in ("identify", "realtime", "guest"):
        raise HTTPException(status_code=400, detail="Source tidak valid")
    supabase.table("detection_logs").insert({
        "name": body.name,
        "similarity": body.similarity,
        "det_score": body.det_score,
        "source": body.source,
        "org_id": session["org_id"],
    }).execute()
    return {"ok": True}


@app.get("/api/logs")
def list_logs(limit: int = 50, session: dict = Depends(get_session)):
    """Riwayat deteksi terbaru, urut dari yang paling baru."""
    require_db()
    limit = max(1, min(limit, 200))
    res = (
        supabase.table("detection_logs")
        .select("id, name, similarity, det_score, source, created_at")
        .eq("org_id", session["org_id"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"logs": res.data}


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
    require_db()
    res = supabase.table("app_settings").select("key, value").eq("org_id", session["org_id"]).execute()
    settings = dict(DEFAULT_SETTINGS)
    for row in res.data:
        if row["key"] in settings:
            settings[row["key"]] = row["value"]
    return settings


@app.put("/api/settings")
def update_settings(body: SettingsRequest, session: dict = Depends(get_session)):
    """Perbarui pengaturan sapaan. Tersimpan per organisasi."""
    require_db()
    org_id = session["org_id"]
    updates = body.model_dump(exclude_none=True)
    for key, value in updates.items():
        supabase.table("app_settings").upsert(
            {"org_id": org_id, "key": key, "value": value},
            on_conflict="org_id,key",
        ).execute()
    return get_settings(session)


@app.get("/api/people")
def list_people(session: dict = Depends(get_session)):
    """Daftar orang terdaftar di org ini, dikelompokkan per nama."""
    require_db()
    res = (
        supabase.table("faces")
        .select("id, name, title, created_at")
        .eq("org_id", session["org_id"])
        .order("created_at", desc=True)
        .execute()
    )
    grouped: dict[str, dict] = {}
    for row in res.data:
        g = grouped.setdefault(row["name"], {
            "name": row["name"],
            "title": row.get("title", ""),
            "photos": 0,
            "entries": [],
        })
        g["photos"] += 1
        g["entries"].append({"id": row["id"], "created_at": row["created_at"]})
    return {"total_entries": len(res.data), "people": list(grouped.values())}


class PersonUpdateRequest(BaseModel):
    new_name: str | None = None
    title: str | None = None


@app.patch("/api/people/{name}")
def update_person(name: str, body: PersonUpdateRequest, session: dict = Depends(get_session)):
    """Ganti nama dan/atau gelar semua entri wajah milik satu orang."""
    require_db()
    update: dict = {}
    if body.new_name is not None:
        new_name = body.new_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Nama tidak boleh kosong")
        update["name"] = new_name
    if body.title is not None:
        if body.title not in ("", "Bapak", "Ibu"):
            raise HTTPException(status_code=400, detail="Gelar tidak valid")
        update["title"] = body.title
    if not update:
        return {"updated_entries": 0}
    res = (
        supabase.table("faces")
        .update(update)
        .eq("name", name)
        .eq("org_id", session["org_id"])
        .execute()
    )
    return {"updated_entries": len(res.data or [])}


@app.delete("/api/faces/{face_id}")
def delete_face(face_id: str, session: dict = Depends(get_session)):
    require_db()
    supabase.table("faces").delete().eq("id", face_id).eq("org_id", session["org_id"]).execute()
    return {"deleted": face_id}


@app.delete("/api/people/{name}")
def delete_person(name: str, session: dict = Depends(get_session)):
    require_db()
    res = supabase.table("faces").delete().eq("name", name).eq("org_id", session["org_id"]).execute()
    return {"deleted_name": name, "deleted_entries": len(res.data or [])}


# ----- Static UI -----

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/sw.js")
def service_worker():
    # Disajikan dari root agar scope service worker mencakup seluruh app
    return FileResponse("static/sw.js", media_type="application/javascript")
