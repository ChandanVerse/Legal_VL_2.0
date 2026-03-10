import { marked } from 'marked'
import Citations from './Citations'
import s from './MessageRow.module.css'

marked.use({ breaks: true, gfm: true })

function ts() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function MessageRow({ message }) {
  const { role, content, sources, turn, typing, error } = message
  const isUser = role === 'user'

  return (
    <div className={[s.row, s[role], error ? s.error : ''].join(' ')}>
      <div className={s.gutter}>
        <span className={s.role}>{isUser ? 'You' : 'Lex'}</span>
        <span className={s.turn}>§ {String(turn).padStart(2, '0')}</span>
      </div>

      <div className={s.body}>
        {typing ? (
          <div className={s.typing}>
            <span /><span /><span />
          </div>
        ) : isUser ? (
          <p className={s.userText}>{content}</p>
        ) : (
          <div
            className={s.markdown}
            dangerouslySetInnerHTML={{ __html: marked.parse(content) }}
          />
        )}

        {!typing && sources?.length > 0 && <Citations sources={sources} />}
        {!typing && <div className={s.time}>{ts()}</div>}
      </div>
    </div>
  )
}
