# Face ID — Identifikasi Orang dari Foto

Aplikasi web face recognition server-side dengan akurasi tinggi.

**Stack:** FastAPI + InsightFace (model buffalo_l) + Supabase pgvector + UI HTML/JS

**Cara kerja:** setiap wajah diubah menjadi *embedding* (vektor 512 angka) oleh
InsightFace. Embedding disimpan di Supabase, lalu pencocokan dilakukan dengan
*cosine similarity* via pgvector — cepat bahkan saat data sudah ribuan wajah.

---

## 1. Persiapan Supabase

1. Buat project baru di [supabase.com](https://supabase.com) (atau pakai project yang ada)
2. Buka **SQL Editor**, salin seluruh isi `supabase_setup.sql`, lalu **Run**
3. Buka **Project Settings → API**, catat:
   - `Project URL`
   - `service_role` key (bagian *Project API keys*, klik *Reveal*)

## 2. Konfigurasi environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```

Isi `.env` dengan URL dan service key dari langkah sebelumnya.

> ⚠️ `service_role` key punya akses penuh ke database — jangan pernah
> di-commit ke Git atau dipakai di frontend. File `.gitignore` sudah
> mengecualikannya.

## 3. Install dependensi

Butuh **Python 3.10–3.12**.

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

**Catatan khusus Windows:** paket `insightface` dikompilasi saat install,
jadi perlu **Microsoft C++ Build Tools**. Kalau muncul error
`Microsoft Visual C++ 14.0 or greater is required`:

1. Download [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Saat install, centang **"Desktop development with C++"**
3. Restart terminal, lalu jalankan `pip install -r requirements.txt` lagi

## 4. Jalankan

```bash
uvicorn main:app --reload
```

Saat pertama kali dijalankan, model **buffalo_l (~280 MB)** otomatis
diunduh ke folder `~/.insightface` — tunggu sampai muncul "Model siap."

Buka **http://localhost:8000**

## 5. Cara pakai

1. Tab **DAFTARKAN** → upload foto + nama. Daftarkan **2–3 foto berbeda
   per orang** (sudut/pencahayaan beda) agar akurasi jauh lebih baik
2. Tab **IDENTIFIKASI** → upload foto apa pun (boleh berisi banyak wajah),
   sistem menandai setiap wajah dengan nama + persentase kemiripan
3. Tab **DATABASE** → lihat dan hapus orang terdaftar

### Tentang threshold

- `0.40` (default) — seimbang, cocok untuk mayoritas kasus
- Naikkan ke `0.45–0.50` kalau sering salah orang (*false positive*)
- Turunkan ke `0.35` kalau orang yang benar sering tidak dikenali

---

## Struktur project

```
face-id/
├── main.py              # API FastAPI + InsightFace
├── static/index.html    # UI web
├── supabase_setup.sql   # Skema database (jalankan sekali di Supabase)
├── requirements.txt
├── .env.example
└── README.md
```

## API endpoints

| Method | Endpoint              | Fungsi                              |
|--------|-----------------------|-------------------------------------|
| POST   | `/api/register`       | Daftarkan wajah (form: name, file)  |
| POST   | `/api/identify`       | Identifikasi semua wajah di foto    |
| GET    | `/api/people`         | Daftar orang terdaftar              |
| DELETE | `/api/people/{name}`  | Hapus semua data satu orang         |
| GET    | `/api/health`         | Status server                       |

## Deployment

Backend ini **tidak bisa** di-deploy ke Vercel (butuh proses Python
yang berjalan terus + model ~280 MB di memori). Opsi yang cocok:

- **Railway / Render** — paling mudah, deploy dari repo Git
- **VPS** (mis. IDCloudHost, Biznet Gio) — paling hemat untuk jangka panjang
- Frontend boleh dipisah ke Next.js/Vercel nanti, tinggal arahkan
  fetch ke URL backend

## Catatan hukum (penting)

Data wajah termasuk **data pribadi sensitif** menurut UU PDP No. 27/2022.
Untuk penggunaan nyata, pastikan ada **persetujuan eksplisit** dari setiap
orang yang didaftarkan, dan sediakan mekanisme penghapusan data
(sudah tersedia di tab DATABASE).
