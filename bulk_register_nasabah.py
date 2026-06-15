"""
Impor massal data nasabah dari folder foto ke tabel 'faces' di Supabase.

Sumber: D:\\1. TELLER\\2.FOTO NASABAH
Ada tiga jenis sumber data:
  1. Subfolder per nasabah/kelompok — nama folder (tanpa awalan
     penomoran seperti "13. ") dipakai sebagai nama. Semua foto di
     dalam folder (termasuk subfolder) dipindai untuk mencari wajah.
  2. Foto lepas di folder utama yang nama filenya adalah nama nasabah
     (mis. "AB. SUPARDIN.jpg") — nama file (tanpa ekstensi) dipakai
     sebagai nama.
  3. File .docx di folder utama yang nama filenya adalah nama nasabah
     (mis. "abang efendi.docx") — gambar yang ditempel di dalam
     dokumen (word/media/*) ikut dipindai untuk mencari wajah.

Untuk tiap nasabah, foto dengan skor deteksi wajah tertinggi yang
dipakai untuk pendaftaran. Yang tidak ada wajah terdeteksi otomatis
dilewati.

Penggunaan:
  python bulk_register_nasabah.py                  # dry run, tulis laporan CSV saja
  python bulk_register_nasabah.py --register        # daftarkan ke Supabase
  python bulk_register_nasabah.py --only docx       # hanya proses sumber file .docx
  python bulk_register_nasabah.py --only media       # hanya proses folder & foto lepas
"""

import argparse
import csv
import os
import re
import sys
import zipfile

import cv2
import numpy as np
from dotenv import load_dotenv

SOURCE_DIR = r"D:\1. TELLER\2.FOTO NASABAH"
REPORT_PATH = "nasabah_import_report.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DOC_EXTS = {".docx"}
MAX_SIDE = 1600

# Nama kelompok/instansi/tempat (bukan nasabah perorangan) yang foto
# wajahnya terdeteksi dari foto acara/kelompok, jadi dilewati saat impor.
EXCLUDE_NAMES = {
    "DESAAAAAAAAAAAAAAAAA",
    "GEREJA",
    "BUMDES",
    "KUMPULAN CU",
    "P3K & PNS",
    "PENSIUNAN",
    "DANA BOS, BOP PAUD& SD&SMP",
    "HOTEL BADAU PERMAI",
    "KELOMPOK PEMUDA BERSATU MAJANG",
    "KORDIK PURING KENCANA",
    "PANITIA HUT RI 2023",
    "PUSKESMAS BADAU",
    "SANGGAR SERAKOP BALAI RUAI",
    "SRT KUASA!!!",
    "TPU TEMAWI TINTING",
    "YAYASAN BUKIT PERAK BDU",
    "GEREMPUNG KITAI SERIANG",
    "APKASINDO KAPUAS HULU",
    "BADAN 1125",
    "BADAN 5082",
    "BALAI ADAT BADAU",
    "IBI RANTING UTARA",
    "PT PALMA",
    "paud kemmantan puring",
    "sd kura",
    "sdn 05 kedang",
    "sdn 06 kantuk bunut",
    "sdn 1 badau",
    "sds tunas sejahtera",
    "smpn 02 satap badau",
    "smpn 1 badau",
}


def clean_name(folder_name: str) -> str:
    return re.sub(r"^\d+\.\s*", "", folder_name).strip()


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def read_image(path: str):
    return decode_image(np.fromfile(path, dtype=np.uint8).tobytes())


def docx_image_members(docx_path: str):
    """Daftar nama anggota gambar (word/media/*) di dalam file .docx."""
    try:
        with zipfile.ZipFile(docx_path) as z:
            return [
                n for n in z.namelist()
                if n.startswith("word/media/") and os.path.splitext(n)[1].lower() in IMAGE_EXTS
            ]
    except Exception:
        return []


def load_candidate(candidate):
    """Muat gambar dari kandidat (("img", path) atau ("docx", docx_path, member))."""
    if candidate[0] == "img":
        return read_image(candidate[1])
    _, docx_path, member = candidate
    try:
        with zipfile.ZipFile(docx_path) as z:
            return decode_image(z.read(member))
    except Exception:
        return None


def candidate_label(candidate):
    if candidate[0] == "img":
        return os.path.basename(candidate[1])
    _, docx_path, member = candidate
    return f"{os.path.basename(docx_path)}::{os.path.basename(member)}"


def build_entries(only: str = "all"):
    """Kembalikan daftar (label, name, daftar_kandidat_gambar)."""
    entries = []
    for d in sorted(os.listdir(SOURCE_DIR)):
        full = os.path.join(SOURCE_DIR, d)
        if os.path.isdir(full):
            if only == "docx":
                continue
            name = clean_name(d)
            candidates = []
            for root, _dirs, fnames in os.walk(full):
                for fname in fnames:
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                        candidates.append(("img", os.path.join(root, fname)))
            entries.append((d, name, candidates))
        else:
            stem, ext = os.path.splitext(d)
            ext = ext.lower()
            if ext in IMAGE_EXTS:
                if only == "docx":
                    continue
                entries.append((d, clean_name(stem), [("img", full)]))
            elif ext in DOC_EXTS:
                if only == "media":
                    continue
                name = clean_name(stem)
                # Nama file sering memuat catatan tambahan dalam kurung, mis.
                # "ALBET(DOMI)" atau "SABIANUS SIGAN(PAULUS MAMBANG)" - buang
                # bagian itu supaya cocok/duplikat dengan nama yang sudah
                # terdaftar dari sumber lain.
                stripped = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
                if stripped:
                    name = stripped
                candidates = [("docx", full, member) for member in docx_image_members(full)]
                entries.append((d, name, candidates))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="Daftarkan ke Supabase (default: dry run)")
    parser.add_argument(
        "--only", choices=["all", "media", "docx"], default="all",
        help="Batasi sumber: 'media' (folder & foto lepas), 'docx' (file .docx), atau 'all' (default)",
    )
    args = parser.parse_args()

    load_dotenv()

    print("Memuat model InsightFace (buffalo_l)...", flush=True)
    from insightface.app import FaceAnalysis

    face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_engine.prepare(ctx_id=0, det_size=(640, 640))
    print("Model siap.", flush=True)

    supabase = None
    existing_names = set()
    if args.register:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            print("SUPABASE_URL / SUPABASE_SERVICE_KEY belum diisi di .env")
            sys.exit(1)
        supabase = create_client(url, key)
        res = supabase.table("faces").select("name").execute()
        existing_names = {row["name"] for row in res.data}
        print(f"{len(existing_names)} nama sudah terdaftar, akan dilewati jika cocok.", flush=True)

    entries = build_entries(args.only)
    print(f"{len(entries)} entri ditemukan ({sum(1 for _, _, f in entries if len(f) > 1)} folder, "
          f"{sum(1 for _, _, f in entries if len(f) == 1)} foto lepas).", flush=True)

    rows = []
    for i, (label, name, candidates) in enumerate(entries, 1):
        if not name:
            print(f"[{i}/{len(entries)}] {label} -> nama kosong, dilewati", flush=True)
            rows.append({"source": label, "name": name, "file": "", "det_score": "", "status": "nama kosong, dilewati"})
            continue

        if name in EXCLUDE_NAMES:
            print(f"[{i}/{len(entries)}] {label} -> bukan nasabah perorangan, dilewati", flush=True)
            rows.append({"source": label, "name": name, "file": "", "det_score": "", "status": "bukan nasabah perorangan, dilewati"})
            continue

        if args.register and name in existing_names:
            print(f"[{i}/{len(entries)}] {label} -> sudah terdaftar, dilewati", flush=True)
            rows.append({"source": label, "name": name, "file": "", "det_score": "", "status": "sudah terdaftar, dilewati"})
            continue

        best = None  # (det_score, label, embedding)
        for candidate in candidates:
            try:
                img = load_candidate(candidate)
                if img is None:
                    continue
                faces = face_engine.get(img)
            except Exception:
                continue
            if not faces:
                continue
            face = max(faces, key=lambda f: f.det_score)
            if best is None or face.det_score > best[0]:
                best = (float(face.det_score), candidate_label(candidate), face.normed_embedding.astype(float).tolist())

        if best is None:
            print(f"[{i}/{len(entries)}] {label} -> tidak ada wajah terdeteksi", flush=True)
            rows.append({"source": label, "name": name, "file": "", "det_score": "", "status": "tidak ada wajah terdeteksi"})
            continue

        det_score, fname, embedding = best
        print(f"[{i}/{len(entries)}] {label} -> {fname} (det_score={det_score:.3f})", flush=True)

        if args.register:
            supabase.table("faces").insert({"name": name, "embedding": embedding}).execute()
            rows.append({"source": label, "name": name, "file": fname, "det_score": f"{det_score:.3f}", "status": "terdaftar"})
        else:
            rows.append({"source": label, "name": name, "file": fname, "det_score": f"{det_score:.3f}", "status": "kandidat (dry run)"})

    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "name", "file", "det_score", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSelesai. Laporan ditulis ke {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
