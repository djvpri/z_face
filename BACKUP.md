# Backup Database ZFace (Railway PostgreSQL)

Database menyimpan data biometrik nasabah (wajah + tanda tangan) semua organisasi.
**Wajib ada backup** agar data tidak hilang permanen.

## Opsi A — Backup otomatis Railway (paling disarankan)

1. Buka Railway Dashboard → service **PostgreSQL** → tab **Backups**.
2. Aktifkan **Scheduled Backups** (tersedia di plan berbayar Railway).
3. Atur jadwal harian + retensi (mis. simpan 7–30 hari terakhir).

Restore: dari tab Backups yang sama, pilih snapshot → Restore.

## Opsi B — Backup manual / terjadwal sendiri (pg_dump)

Butuh `pg_dump` (ikut paket PostgreSQL client). Ambil `DATABASE_URL` dari
Railway → PostgreSQL → Variables (jangan commit ke git).

### Sekali jalan
```bash
pg_dump "POSTEGRES_CONNECTION_URL" -Fc -f zface_backup_$(date +%Y%m%d).dump
```

### Restore dari file dump
```bash
pg_restore --clean --if-exists -d "POSTEGRES_CONNECTION_URL" zface_backup_YYYYMMDD.dump
```

### Windows PowerShell (terjadwal harian via Task Scheduler)
```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmm"
& "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" $env:ZFACE_DB_URL -Fc -f "D:\backup\zface_$ts.dump"
```
Set `ZFACE_DB_URL` sebagai environment variable (jangan tulis URL langsung di skrip),
lalu jadwalkan skrip ini lewat **Task Scheduler** tiap hari.

## Skrip siap pakai + rotasi otomatis (Windows)

Repo ini punya `backup_zface.ps1` — dump otomatis + hapus file lama + log.

**1. Install PostgreSQL client** (untuk `pg_dump`): https://www.postgresql.org/download/windows/
   Sesuaikan path `$PgDump` di skrip dengan versi yang ter-install.

**2. Set kredensial sebagai environment variable (sekali saja, jangan di skrip):**
```powershell
setx ZFACE_DB_URL "POSTGRES_CONNECTION_URL_DARI_RAILWAY"
```
Tutup & buka ulang PowerShell setelah `setx`.

**3. Uji jalankan manual:**
```powershell
powershell -ExecutionPolicy Bypass -File "D:\ANJUV\z_face\face-id\face-id\backup_zface.ps1"
```
Cek folder `D:\backup\zface` — harus ada file `.dump` dan `backup.log`.

**4. Jadwalkan harian via Task Scheduler:**
- Buka **Task Scheduler** → Create Basic Task → trigger **Daily** (mis. jam 01:00)
- Action: **Start a program**
  - Program: `powershell.exe`
  - Arguments: `-ExecutionPolicy Bypass -File "D:\ANJUV\z_face\face-id\face-id\backup_zface.ps1"`
- Centang "Run whether user is logged on or not" agar tetap jalan walau belum login.

Atur retensi & folder lewat `$RetentionDays` / `$BackupDir` di dalam skrip.

## Catatan keamanan
- File dump berisi data sensitif (embedding biometrik, hash password). Simpan
  terenkripsi / di lokasi terbatas, jangan di folder publik atau repo git.
- Uji **restore** secara berkala — backup yang tidak pernah diuji = belum tentu bisa dipakai.
