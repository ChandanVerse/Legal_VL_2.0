import { useState, useRef, useCallback } from 'react'
import * as api from '../api'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [busy, setBusy]         = useState(false)

  const idRef        = useRef(0)
  const turnRef      = useRef(0)
  const sessionIdRef = useRef(null)
  const busyRef      = useRef(false)

  const nextId   = () => ++idRef.current
  const nextTurn = () => ++turnRef.current

  async function ensureSession() {
    let sid = sessionIdRef.current
    if (!sid) {
      sid = await api.newSession()
      sessionIdRef.current = sid
    }
    return sid
  }

  const newSession = useCallback(() => {
    if (busyRef.current) return
    const old = sessionIdRef.current
    sessionIdRef.current = null
    setMessages([])
    turnRef.current = 0
    if (old) api.deleteSession(old)
  }, [])

  const sendMessage = useCallback(async (text) => {
    if (busyRef.current || !text.trim()) return

    let sid
    try { sid = await ensureSession() } catch (e) {
      setMessages([{ id: nextId(), role: 'error', content: e.message, turn: 0 }])
      return
    }

    const turn   = nextTurn()
    const userId = nextId()
    const asstId = nextId()

    setMessages(prev => [
      ...prev,
      { id: userId, role: 'user',      content: text, turn },
      { id: asstId, role: 'assistant', content: '',   turn, typing: true },
    ])
    setBusy(true)
    busyRef.current = true

    try {
      const data = await api.sendMessage(sid, text)
      setMessages(prev => prev.map(m =>
        m.id === asstId
          ? { ...m, content: data.reply, sources: data.sources ?? [], typing: false }
          : m
      ))
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === asstId
          ? { ...m, content: e.message, typing: false, error: true }
          : m
      ))
    } finally {
      setBusy(false)
      busyRef.current = false
    }
  }, [])

  const sendPdfQuery = useCallback(async (file, prompt = '') => {
    if (busyRef.current) return

    let sid
    try { sid = await ensureSession() } catch (e) {
      setMessages([{ id: nextId(), role: 'error', content: e.message, turn: 0 }])
      return
    }

    const turn   = nextTurn()
    const userId = nextId()
    const asstId = nextId()

    const displayText = prompt
      ? `${prompt} [Uploaded: ${file.name}]`
      : `Find similar cases for: ${file.name}`

    setMessages(prev => [
      ...prev,
      { id: userId, role: 'user',      content: displayText, turn },
      { id: asstId, role: 'assistant', content: '',           turn, typing: true },
    ])
    setBusy(true)
    busyRef.current = true

    try {
      const data = await api.queryWithPdf(sid, file, prompt)
      setMessages(prev => prev.map(m =>
        m.id === asstId
          ? { ...m, content: data.reply, sources: data.sources ?? [], typing: false }
          : m
      ))
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === asstId
          ? { ...m, content: e.message, typing: false, error: true }
          : m
      ))
    } finally {
      setBusy(false)
      busyRef.current = false
    }
  }, [])

  const cleanup = useCallback(() => {
    if (sessionIdRef.current) api.deleteSession(sessionIdRef.current)
  }, [])

  return { messages, busy, newSession, sendMessage, sendPdfQuery, cleanup }
}
