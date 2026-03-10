import { useEffect, useRef } from 'react'
import MessageRow from './MessageRow'
import ScalesIcon from './ScalesIcon'
import s from './ChatPane.module.css'

export default function ChatPane({ messages }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (!messages.length) {
    return (
      <div className={s.empty}>
        <ScalesIcon size={56} className={s.emptyIcon} />
        <h1 className={s.emptyTitle}>Awaiting Query</h1>
        <div className={s.emptyRule} />
        <p className={s.emptySub}>
          Open a new session and pose your question.
          <br />
          Relevant Supreme Court judgments will be retrieved automatically.
        </p>
      </div>
    )
  }

  return (
    <div className={s.pane}>
      {messages.map(msg => (
        <MessageRow key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
