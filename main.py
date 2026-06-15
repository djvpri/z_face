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
            }
        )

    return {"image": {"width": w, "height": h}, "faces": results}


@app.get("/api/people")
def list_people():
    """Daftar orang terdaftar, dikelompokkan per nama."""
    require_db()
    res = (
        supabase.table("faces")
        .select("id, name, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    grouped: dict[str, dict] = {}
    for row in res.data:
        g = grouped.setdefault(row["name"], {"name": row["name"], "photos": 0, "entries": []})
        g["photos"] += 1
        g["entries"].append({"id": row["id"], "created_at": row["created_at"]})
    return {"total_entries": len(res.data), "people": list(grouped.values())}


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
