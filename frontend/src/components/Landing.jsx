import { useRef, useState } from 'react'
import ScalesIcon from './ScalesIcon'
import AttachButton from './AttachButton'
import s from './Landing.module.css'

export default function Landing({ onSend, busy, onPdfQuery }) {
  const ref = useRef(null)
  const [attachedFile, setAttachedFile] = useState(null)

  function autoResize() {
    const el = ref.current
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 140) + 'px'
  }

  function submit() {
    const text = ref.current.value.trim()
    if (attachedFile) {
      onPdfQuery(attachedFile, text)
      setAttachedFile(null)
      ref.current.value = ''
      autoResize()
      return
    }
    if (!text || busy) return
    ref.current.value = ''
    autoResize()
    onSend(text)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className={s.landing}>
      <div className={s.top}>
        <ScalesIcon size={52} className={s.icon} />
        <div className={s.title}>Lex</div>
      </div>
      <div className={s.bottom}>
        <div className={[s.frame, busy ? s.inactive : ''].join(' ')}>
          <AttachButton
            onAttach={setAttachedFile}
            attachedFile={attachedFile}
            disabled={busy}
          />
          <textarea
            ref={ref}
            rows={1}
            className={s.input}
            placeholder={attachedFile ? `${attachedFile.name} attached — add a prompt or just send` : 'Enter your legal query…'}
            disabled={busy}
            onInput={autoResize}
            onKeyDown={handleKeyDown}
            autoFocus
          />
          <button
            className={s.sendBtn}
            disabled={busy}
            onClick={submit}
            title="Submit query"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        </div>
        <div className={s.hint}>↵ Submit · ⇧↵ Newline</div>
      </div>
    </div>
  )
}
