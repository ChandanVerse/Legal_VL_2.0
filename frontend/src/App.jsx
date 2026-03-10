import { useEffect } from 'react'
import { useState } from 'react'
import { useChat } from './hooks/useChat'
import Rail from './components/Rail'
import ChatPane from './components/ChatPane'
import InputBar from './components/InputBar'
import Landing from './components/Landing'
import s from './App.module.css'

export default function App() {
  const { messages, busy, newSession, sendMessage, sendPdfQuery, cleanup } = useChat()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => {
    window.addEventListener('beforeunload', cleanup)
    return () => {
      window.removeEventListener('beforeunload', cleanup)
      cleanup()
    }
  }, [cleanup])

  return (
    <div className={s.layout}>
      <Rail
        busy={busy}
        onNewSession={newSession}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(o => !o)}
      />

      <main className={s.chamber}>
        {messages.length === 0 ? (
          <Landing onSend={sendMessage} busy={busy}
                   onPdfQuery={sendPdfQuery} />
        ) : (
          <>
            <div className={s.goldRule} />

            <div className={s.transcript}>
              <ChatPane messages={messages} />
            </div>

            <InputBar busy={busy} disabled={false} onSend={sendMessage}
                      onPdfQuery={sendPdfQuery} />
          </>
        )}
      </main>
    </div>
  )
}
