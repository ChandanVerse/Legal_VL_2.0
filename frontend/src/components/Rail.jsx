import ScalesIcon from './ScalesIcon'
import s from './Rail.module.css'

export default function Rail({ busy, onNewSession, open, onToggle }) {
  return (
    <aside className={[s.rail, open ? '' : s.collapsed].join(' ')}>
      {open ? (
        <>
          <div className={s.brand}>
            <ScalesIcon size={28} className={s.brandIcon} />
            <div className={s.brandName}>Lex</div>
          </div>

          <div className={s.actions}>
            <button className={s.newBtn} onClick={onNewSession} disabled={busy}>
              <span className={s.newDot} />
              New Chat
            </button>
          </div>

          <div className={s.spacer} />

          <div className={s.footer}>
            <button className={s.collapseBtn} onClick={onToggle} title="Collapse sidebar">◀</button>
          </div>
        </>
      ) : (
        <div className={s.strip}>
          <ScalesIcon size={18} className={s.stripIcon} />
          <button className={s.stripBtn} onClick={onNewSession} disabled={busy} title="New Chat">+</button>
          <button className={s.stripToggle} onClick={onToggle} title="Expand">▶</button>
        </div>
      )}
    </aside>
  )
}
