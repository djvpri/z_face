'use client'

import { useSession, signOut } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

type FaceStatus = {
  hasFace: boolean
  facesCount: number
  faces: Array<{
    id: string
    label: string | null
    quality: number
    isPrimary: boolean
    createdAt: string
  }>
  recentLogs: Array<{
    id: string
    action: string
    confidence: number | null
    createdAt: string
  }>
}

export default function DashboardPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [faceStatus, setFaceStatus] = useState<FaceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [registering, setRegistering] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraFacing, setCameraFacing] = useState<'user' | 'environment'>('user')
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (status === 'unauthenticated') router.push('/login')
  }, [status, router])

  useEffect(() => {
    if (session) {
      fetch('/api/face/status')
        .then(r => r.json())
        .then(setFaceStatus)
        .finally(() => setLoading(false))
    }
  }, [session])

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const faceapi = await import('face-api.js')
        const MODEL_URL = '/models'
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ])
        setModelsLoaded(true)
      } catch (err) {
        console.error('Failed to load face models:', err)
      }
    }
    loadModels()
  }, [])

  // Register face
  const handleRegisterFace = async () => {
    setRegistering(true)
    setMessage('')
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: cameraFacing }
      })
      
      const video = document.createElement('video')
      video.srcObject = stream
      video.autoplay = true
      video.playsInline = true
      
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const faceapi = await import('face-api.js')
      
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor()
      
      stream.getTracks().forEach(t => t.stop())
      
      if (!detection) {
        setMessage('Wajah tidak terdeteksi. Coba lagi dengan pencahayaan yang baik.')
        setRegistering(false)
        return
      }
      
      const embedding = Array.from(detection.descriptor)
      
      const res = await fetch('/api/face/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embeddings: [embedding],
          label: 'Primary',
          quality: 0.8,
        }),
      })
      
      const data = await res.json()
      
      if (data.success) {
        setMessage(`✅ Wajah berhasil didaftarkan! (${data.embeddingsCount} embedding)`)
        // Refresh status
        const statusRes = await fetch('/api/face/status')
        setFaceStatus(await statusRes.json())
      } else {
        setMessage('❌ Gagal mendaftarkan wajah')
      }
    } catch (err) {
      setMessage('❌ Error: ' + (err as Error).message)
    } finally {
      setRegistering(false)
    }
  }

  // Delete face
  const handleDeleteFace = async () => {
    if (!confirm('Hapus semua data wajah?')) return
    
    try {
      const res = await fetch('/api/face/register', { method: 'DELETE' })
      const data = await res.json()
      
      if (data.success) {
        setMessage('✅ Data wajah dihapus')
        setFaceStatus({ hasFace: false, facesCount: 0, faces: [], recentLogs: [] })
      }
    } catch (err) {
      setMessage('❌ Gagal menghapus data wajah')
    }
  }

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!session) return null

  const user = session.user as any

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center shadow-lg">
              <span className="text-white font-bold">Z</span>
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight text-white">Z Face</h1>
              <p className="text-xs text-slate-400">Face Recognition Service</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <div className="text-sm font-medium text-white">{user.name}</div>
              <div className="text-xs text-slate-400">{user.role}</div>
            </div>
            <button onClick={() => signOut({ callbackUrl: '/login' })}
              className="text-slate-400 hover:text-red-400 transition-colors p-2">
              🚪
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
        {/* Welcome */}
        <div className="mb-8">
          <h2 className="text-xl sm:text-2xl font-bold text-white mb-1">Halo, {user.name?.split(' ')[0]} 👋</h2>
          <p className="text-slate-400 text-sm">Kelola data wajah Anda untuk login</p>
        </div>

        {/* Message */}
        {message && (
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 mb-6 text-white">
            {message}
          </div>
        )}

        {/* Face Status Card */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 mb-6 backdrop-blur-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
              <span className="text-2xl">🔐</span>
            </div>
            <div>
              <h3 className="font-semibold text-white">Status Wajah</h3>
              <p className="text-sm text-slate-400">
                {faceStatus?.hasFace ? '✅ Wajah terdaftar' : '⚠️ Belum ada wajah terdaftar'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <div className="bg-slate-900/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Total Embedding</div>
              <div className="text-xl font-bold text-purple-400">{faceStatus?.facesCount || 0}</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Login Terakhir</div>
              <div className="text-sm font-medium text-white">
                {faceStatus?.recentLogs.find(l => l.action === 'LOGIN')
                  ? new Date(faceStatus.recentLogs.find(l => l.action === 'LOGIN')!.createdAt).toLocaleDateString('id-ID')
                  : '-'}
              </div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Models</div>
              <div className="text-sm font-medium text-white">{modelsLoaded ? '✅ Loaded' : '⏳ Loading...'}</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Percobaan Gagal</div>
              <div className="text-xl font-bold text-red-400">
                {faceStatus?.recentLogs.filter(l => l.action === 'FAILED').length || 0}
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setCameraFacing(cameraFacing === 'user' ? 'environment' : 'user')}
              className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-3 py-2.5 transition-colors text-sm"
              title={cameraFacing === 'user' ? 'Ganti ke kamera belakang' : 'Ganti ke kamera depan'}
            >
              {cameraFacing === 'user' ? '📱' : '📷'}
            </button>
            <button
              onClick={handleRegisterFace}
              disabled={registering || !modelsLoaded}
              className="flex-1 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-2.5 transition-colors"
            >
              {registering ? '⏳ Mendaftarkan...' : faceStatus?.hasFace ? '🔄 Update Wajah' : '📷 Daftarkan Wajah'}
            </button>
            {faceStatus?.hasFace && (
              <button
                onClick={handleDeleteFace}
                className="bg-red-600/20 hover:bg-red-600/30 text-red-400 font-medium rounded-lg px-4 py-2.5 transition-colors"
              >
                🗑️ Hapus
              </button>
            )}
          </div>
        </div>

        {/* How it works */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
          <h3 className="font-semibold text-white mb-4">📖 Cara Kerja</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-blue-400">1</span>
              </div>
              <div>
                <div className="font-medium text-white">Daftarkan Wajah</div>
                <div className="text-sm text-slate-400">Klik "Daftarkan Wajah" dan biarkan kamera mendeteksi wajah Anda</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-purple-400">2</span>
              </div>
              <div>
                <div className="font-medium text-white">Face Login</div>
                <div className="text-sm text-slate-400">Di aplikasi Z lainnya, pilih "Login dengan Wajah" dan biarkan kamera membaca wajah Anda</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-green-400">3</span>
              </div>
              <div>
                <div className="font-medium text-white">Terkoneksi Otomatis</div>
                <div className="text-sm text-slate-400">Satu wajah bisa login ke semua aplikasi Z (ZOne, ZGold, ZBengkel, ZLaundry, ZResto)</div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-xs text-slate-600">
          Z Face · Face Recognition Service · 2026
        </div>
      </main>
    </div>
  )
}
