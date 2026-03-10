export async function newSession() {
  const res = await fetch('/chat/new', { method: 'POST' })
  if (!res.ok) throw new Error('Failed to create session')
  const data = await res.json()
  return data.session_id
}

export async function sendMessage(sessionId, content) {
  const res = await fetch(`/chat/${sessionId}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export async function queryWithPdf(sessionId, file, prompt = '') {
  const form = new FormData()
  form.append('file', file)
  form.append('prompt', prompt)
  const res = await fetch(`/chat/${sessionId}/pdf-query`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json() // { reply, sources }
}

export function deleteSession(sessionId) {
  return fetch(`/chat/${sessionId}`, { method: 'DELETE', keepalive: true }).catch(() => {})
}
