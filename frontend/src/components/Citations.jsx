import { useState } from 'react'
import s from './Citations.module.css'

function confClass(c) {
  if (c >= 0.75) return s.hi
  if (c >= 0.45) return s.mid
  return s.lo
}

export default function Citations({ sources }) {
  const [open, setOpen] = useState(false)

  return (
    <div className={s.wrap}>
      <button
        className={`${s.btn} ${open ? s.open : ''}`}
        onClick={() => setOpen(o => !o)}
      >
        <span className={s.caret}>▶</span>
        {sources.length} citation{sources.length > 1 ? 's' : ''}
      </button>

      {open && (
        <div className={s.panel}>
          <div className={s.panelHd}>Retrieved Judgments</div>
          {sources.map((src, i) => {
            const pages = src.pages.length > 1
              ? `pp. ${src.pages[0]}–${src.pages[src.pages.length - 1]}`
              : `p. ${src.pages[0]}`
            return (
              <div key={src.case} className={s.item}>
                <span className={s.caseName}>{src.case}</span>
                <span className={s.pages}>{pages}</span>
                <span className={`${s.conf} ${confClass(src.confidence)}`}>
                  {Math.round(src.confidence * 100)}%
                </span>
                <a
                  href={`/files/${encodeURIComponent(src.case)}.pdf`}
                  download={`${src.case}.pdf`}
                  className={s.download}
                  title="Download judgment PDF"
                >↓</a>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
