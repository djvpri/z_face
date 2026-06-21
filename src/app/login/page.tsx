'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { signIn } from 'next-auth/react'
import { useRouter } from 'next/navigation'

type LoginMode = 'password' | 'face'

export default function LoginPage() {
  const [mode, setMode] = useState<LoginMode>('password')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraFacing, setCameraFacing] = useState<'user' | 'environment'>('user')
  const [faceStatus, setFaceStatus] = useState('')
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const router = useRouter()

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        // Dynamically import face-api.js
        const faceapi = await import('face-api.js')
        
        const MODEL_URL = '/models'
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ])
        
        setModelsLoaded(true)
        console.log('Face detection models loaded')
      } catch (err) {
        console.error('Failed to load face models:', err)
      }
    }
    
    loadModels()
  }, [])

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: cameraFacing }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setCameraActive(true)
      }
    } catch (err) {
      setError('Tidak bisa mengakses kamera')
    }
  }, [])

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setCameraActive(false)
  }, [])

  // Detect face and get embedding
  const detectFace = useCallback(async (): Promise<number[] | null> => {
    if (!videoRef.current || !canvasRef.current || !modelsLoaded) return null
    
    const faceapi = await import('face-api.js')
    const video = videoRef.current
    const canvas = canvasRef.current
    
    // Match canvas size to video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    // Detect face
    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor()
    
    if (!detection) return null
    
    // Return 128-dimensional embedding
    return Array.from(detection.descriptor)
  }, [modelsLoaded])

  // Face login
  const handleFaceLogin = async () => {
    if (!email) {
      setError('Masukkan email terlebih dahulu')
      return
    }
    
    setLoading(true)
    setError('')
    setFaceStatus('Mengaktifkan kamera...')
    
    await startCamera()
    
    // Wait for camera to be ready
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    setFaceStatus('Mendeteksi wajah...')
    
    // Try to detect face
    let attempts = 0
    const maxAttempts = 10
    
    const detectLoop = async () => {
      const embedding = await detectFace()
      
      if (embedding) {
        setFaceStatus('Memverifikasi wajah...')
        
        // Send to API
        const res = await fetch('/api/face/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, embedding, threshold: 0.6 }),
        })
        
        const data = await res.json()
        
        if (data.match) {
          setFaceStatus(`Wajah terverifikasi! Confidence: ${(data.confidence * 100).toFixed(1)}%`)
          stopCamera()
          
          // Sign in with credentials
          const signInResult = await signIn('credentials', {
            email,
            password: data.email, // Use email as password placeholder for face login
            redirect: false,
          })
          
          if (signInResult?.ok) {
            router.push('/dashboard')
          } else {
            setError('Login gagal')
            setFaceStatus('')
          }
        } else {
          attempts++
          if (attempts >= maxAttempts) {
            setError('Wajah tidak cocok atau tidak terdaftar')
            setFaceStatus('')
            stopCamera()
          } else {
            setFaceStatus(`Percobaan ${attempts}/${maxAttempts}... Coba lagi`)
            setTimeout(detectLoop, 1000)
          }
        }
      } else {
        attempts++
        if (attempts >= maxAttempts) {
          setError('Wajah tidak terdeteksi')
          setFaceStatus('')
          stopCamera()
        } else {
          setFaceStatus(`Wajah tidak terdeteksi... (${attempts}/${maxAttempts})`)
          setTimeout(detectLoop, 500)
        }
      }
    }
    
    detectLoop()
  }

  // Password login
  const handlePasswordLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    
    const res = await signIn('credentials', {
      email,
      password,
      redirect: false,
    })
    
    setLoading(false)
    
    if (res?.error) {
      setError('Email atau password salah')
    } else {
      router.push('/dashboard')
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 mb-4 shadow-lg shadow-blue-500/25">
            <span className="text-3xl font-bold text-white">Z</span>
          </div>
          <h1 className="text-2xl font-bold text-white">Z Face</h1>
          <p className="text-slate-400 mt-1">Face Recognition Service</p>
        </div>

        {/* Mode Toggle */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => { setMode('password'); stopCamera(); setFaceStatus('') }}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
              mode === 'password'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            🔑 Password
          </button>
          <button
            onClick={() => { setMode('face'); setError('') }}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
              mode === 'face'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            📷 Face
          </button>
        </div>

        {/* Login Form */}
        <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg px-4 py-2 mb-4">
              {error}
            </div>
          )}

          {mode === 'password' ? (
            <form onSubmit={handlePasswordLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="admin@zone.id"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">Password</label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="••••••••"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-2.5 transition-colors"
              >
                {loading ? 'Masuk...' : 'Masuk'}
              </button>
            </form>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="admin@zone.id"
                />
              </div>

              {/* Camera Preview */}
              <div className="relative aspect-video bg-slate-800 rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!cameraActive && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <span className="text-4xl mb-2 block">📷</span>
                      <p className="text-slate-400 text-sm">Kamera belum aktif</p>
                    </div>
                  </div>
                )}
                
                {faceStatus && (
                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-sm p-2 text-center">
                    {faceStatus}
                  </div>
                )}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={async () => { 
                    const newFacing = cameraFacing === 'user' ? 'environment' : 'user'
                    setCameraFacing(newFacing)
                    if (cameraActive) { 
                      streamRef.current?.getTracks().forEach(t => t.stop())
                      try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: newFacing } })
                        if (videoRef.current) {
                          videoRef.current.srcObject = stream
                          streamRef.current = stream
                        }
                      } catch {}
                    }
                  }}
                  className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-3 py-2.5 transition-colors text-sm"
                  title={cameraFacing === 'user' ? 'Ganti kamera belakang' : 'Ganti kamera depan'}
                >
                  {cameraFacing === 'user' ? '📱' : '📷'}
                </button>
                <button
                  onClick={handleFaceLogin}
                  disabled={loading || !modelsLoaded}
                  className="flex-1 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-2.5 transition-colors"
                >
                  {!modelsLoaded ? 'Loading models...' : loading ? 'Memproses...' : '📷 Login dengan Wajah'}
                </button>
              </div>
            </div>
          )}

          <p className="text-center text-xs text-slate-500 mt-4">
            Demo: admin@zone.id / admin123
          </p>
        </div>
      </div>
    </div>
  )
}
