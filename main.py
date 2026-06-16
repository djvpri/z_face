"""
Face ID API — Identifikasi orang dari foto
Stack: FastAPI + InsightFace (buffalo_l) + Supabase pgvector
Jalankan: uvicorn main:app --reload
"""

import os

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- Model InsightFace -----
# Model buffalo_l (~280 MB) otomatis diunduh ke ~/.insightface saat pertama kali dijalankan
print("Memuat model InsightFace (buffalo_l)...")
from insightface.app import FaceAnalysis  # noqa: E402

face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_engine.prepare(ctx_id=0, det_size=(640, 640))
print("Model siap.")

app = FastAPI(title="Face ID API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Helper -----

def require_db():
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase belum dikonfigurasi. Isi SUPABASE_URL dan SUPABASE_SERVICE_KEY di file .env",
        )


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File kosong")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid (gunakan JPG/PNG)")
    # Batasi resolusi agar deteksi cepat dan konsisten
    max_side = 1600
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_faces(img: np.ndarray):
    faces = face_engine.get(img)
    return faces


def largest_face(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    return max(faces, key=area)


# ----- Endpoints -----

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "buffalo_l",
        "database": "terhubung" if supabase else "belum dikonfigurasi",
    }


@app.post("/api/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """Daftarkan wajah baru. Satu orang boleh didaftarkan beberapa kali
    dengan foto berbeda untuk meningkatkan akurasi."""
    require_db()
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong")

    img = await read_image(file)
    faces = detect_faces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="Tidak ada wajah terdeteksi di foto")

    face = largest_face(faces)
    embedding = face.normed_embedding.astype(float).tolist()

    res = (
        supabase.table("faces")
        .insert({"name": name, "embedding": embedding})
        .execute()
    )

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
def register_embedding(body: EmbeddingRegisterRequest):
    """Daftarkan wajah dari embedding yang sudah dihitung sebelumnya
    (misalnya hasil dari /api/identify), tanpa deteksi ulang."""
    require_db()
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong")
    if len(body.embedding) != 512:
        raise HTTPException(status_code=400, detail="Embedding tidak valid")

    res = (
        supabase.table("faces")
        .insert({"name": name, "embedding": body.embedding})
        .execute()
    )

    return {"id": res.data[0]["id"], "name": name}


@app.post("/api/identify")
async def identify(file: UploadFile = File(...), threshold: float = Form(0.40)):
    """Identifikasi semua wajah dalam foto.
    threshold = batas minimum cosine similarity (0.35–0.50 umumnya bagus)."""
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
            },
        ).execute()

        matches = [
            {"id": m["id"], "name": m["name"], "similarity": round(m["similarity"], 4)}
            for m in (rpc.data or [])
        ]

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        results.append(
            {
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "det_score": round(float(face.det_score), 3),
                "matches": matches,
                "best": matches[0] if matches else None,
                "embedding": embedding,
            }
        )

    return {"image": {"width": w, "height": h}, "faces": results}


class LogEntryRequest(BaseModel):
    name: str | None = None
    similarity: float | None = None
    det_score: float | None = None
    source: str


@app.post("/api/logs")
def create_log(body: LogEntryRequest):
    """Catat satu hasil deteksi (dikenal maupun tidak dikenal) ke riwayat."""
    require_db()
    if body.source not in ("identify", "realtime", "guest"):
        raise HTTPException(status_code=400, detail="Source tidak valid")

    supabase.table("detection_logs").insert(
        {
            "name": body.name,
            "similarity": body.similarity,
            "det_score": body.det_score,
            "source": body.source,
        }
    ).execute()

    return {"ok": True}


@app.get("/api/logs")
def list_logs(limit: int = 50):
    """Riwayat deteksi terbaru, urut dari yang paling baru."""
    require_db()
    limit = max(1, min(limit, 200))
    res = (
        supabase.table("detection_logs")
        .select("id, name, similarity, det_score, source, created_at")
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
def get_settings():
    """Ambil pengaturan aplikasi (mis. kalimat sapaan mode TAMU)."""
    require_db()
    res = supabase.table("app_settings").select("key, value").execute()
    settings = dict(DEFAULT_SETTINGS)
    for row in res.data:
        if row["key"] in settings:
            settings[row["key"]] = row["value"]
    return settings


@app.put("/api/settings")
def update_settings(body: SettingsRequest):
    """Perbarui pengaturan aplikasi. Hanya field yang dikirim yang diubah,
    dan tersimpan di server sehingga berlaku sama di semua perangkat."""
    require_db()
    updates = body.model_dump(exclude_none=True)
    for key, value in updates.items():
        supabase.table("app_settings").upsert({"key": key, "value": value}).execute()
    return get_settings()


@app.get("/api/people")
def list_people():
    """Daftar orang terdaftar, dikelompokkan per nama."""
    require_db()
    res = (
        supabase.table("faces")
        .select("id, name, title, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    grouped: dict[str, dict] = {}
    for row in res.data:
        g = grouped.setdefault(row["name"], {"name": row["name"], "title": row.get("title", ""), "photos": 0, "entries": []})
        g["photos"] += 1
        g["entries"].append({"id": row["id"], "created_at": row["created_at"]})
    return {"total_entries": len(res.data), "people": list(grouped.values())}


class PersonUpdateRequest(BaseModel):
    new_name: str | None = None
    title: str | None = None


@app.patch("/api/people/{name}")
def update_person(name: str, body: PersonUpdateRequest):
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
    res = supabase.table("faces").update(update).eq("name", name).execute()
    return {"updated_entries": len(res.data or [])}


@app.delete("/api/faces/{face_id}")
def delete_face(face_id: str):
    require_db()
    supabase.table("faces").delete().eq("id", face_id).execute()
    return {"deleted": face_id}


@app.delete("/api/people/{name}")
def delete_person(name: str):
    require_db()
    res = supabase.table("faces").delete().eq("name", name).execute()
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
