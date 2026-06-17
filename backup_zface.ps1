# ====================================================================
# Backup terjadwal ZFace (Railway PostgreSQL) untuk Windows
# - Dump database ke file ber-timestamp
# - Rotasi otomatis: hapus dump lebih tua dari $RetentionDays
# - Catat hasil ke backup.log
#
# Kredensial TIDAK ditulis di sini. Ambil dari environment variable
# ZFACE_DB_URL (lihat cara set di bawah file ini / di BACKUP.md).
# ====================================================================

# ---- Konfigurasi (sesuaikan) ----
$BackupDir     = "D:\backup\zface"                               # folder tujuan backup
$RetentionDays = 14                                              # simpan berapa hari
$PgDump        = "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" # sesuaikan versi PostgreSQL
# ----------------------------------

$DbUrl = $env:ZFACE_DB_URL

if (-not $DbUrl) {
    Write-Error "Environment variable ZFACE_DB_URL belum di-set. Lihat petunjuk di BACKUP.md."
    exit 1
}
if (-not (Test-Path $PgDump)) {
    Write-Error "pg_dump tidak ditemukan di '$PgDump'. Install PostgreSQL client atau perbaiki path."
    exit 1
}
if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
}

$ts   = Get-Date -Format "yyyyMMdd_HHmmss"
$file = Join-Path $BackupDir "zface_$ts.dump"
$log  = Join-Path $BackupDir "backup.log"

# -Fc = format custom (terkompresi, bisa di-pg_restore selektif)
& $PgDump $DbUrl -Fc -f $file

if ($LASTEXITCODE -eq 0 -and (Test-Path $file)) {
    $mb = [math]::Round((Get-Item $file).Length / 1MB, 2)
    "$(Get-Date -Format s) OK   -> $file ($mb MB)" | Add-Content -Path $log
    # Rotasi: hapus dump lama
    Get-ChildItem $BackupDir -Filter "zface_*.dump" |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays) } |
        ForEach-Object {
            Remove-Item $_.FullName -Force
            "$(Get-Date -Format s) HAPUS lama -> $($_.Name)" | Add-Content -Path $log
        }
} else {
    "$(Get-Date -Format s) GAGAL (exit $LASTEXITCODE)" | Add-Content -Path $log
    exit 1
}
