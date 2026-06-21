import { NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { parseEmbeddings, findBestMatch, FaceEmbedding } from '@/lib/face-utils'

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { email, embedding, threshold } = body

    if (!embedding || !Array.isArray(embedding) || embedding.length !== 128) {
      return NextResponse.json({ error: 'Valid embedding (128-dim) required' }, { status: 400 })
    }

    // Find user(s) — email optional for face-only login
    let users
    if (email) {
      const user = await prisma.user.findUnique({
        where: { email },
        include: { faces: true },
      })
      users = user ? [user] : []
    } else {
      // Face-only login: search ALL users
      users = await prisma.user.findMany({ include: { faces: true } })
    }

    if (users.length === 0) {
      return NextResponse.json({ match: false, error: 'User not found' }, { status: 404 })
    }

    // Find best match across all users
    let bestResult = { match: false, confidence: 0, distance: 1, userId: '', name: '', email: '', role: '' }
    
    for (const user of users) {
      if (!user.faces || user.faces.length === 0) continue

      const storedEmbeddings: FaceEmbedding[] = []
      for (const face of user.faces) {
        const embs = parseEmbeddings(face.embeddings)
        storedEmbeddings.push(...embs)
      }

      if (storedEmbeddings.length === 0) continue

      const result = findBestMatch(embedding, storedEmbeddings, threshold || 0.5)

      if (result.match && result.confidence > bestResult.confidence) {
        bestResult = {
          match: true,
          confidence: result.confidence,
          distance: result.distance,
          userId: user.id,
          name: user.name,
          email: user.email,
          role: user.role,
        }
      }
    }

    // Log attempt
    if (bestResult.match) {
      await prisma.faceLog.create({
        data: {
          userId: bestResult.userId,
          action: 'LOGIN',
          confidence: bestResult.confidence,
          deviceInfo: req.headers.get('user-agent') || undefined,
          ipAddress: req.headers.get('x-forwarded-for') || undefined,
        },
      })

      return NextResponse.json({
        match: true,
        confidence: bestResult.confidence,
        distance: bestResult.distance,
        userId: bestResult.userId,
        name: bestResult.name,
        email: bestResult.email,
        role: bestResult.role,
      })
    } else {
      // Log failed attempt for first user
      if (users.length > 0) {
        await prisma.faceLog.create({
          data: {
            userId: users[0].id,
            action: 'FAILED',
            confidence: 0,
            deviceInfo: req.headers.get('user-agent') || undefined,
            ipAddress: req.headers.get('x-forwarded-for') || undefined,
          },
        })
      }

      return NextResponse.json({
        match: false,
        confidence: 0,
        distance: 1,
      })
    }
  } catch (error) {
    console.error('Face verify error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
