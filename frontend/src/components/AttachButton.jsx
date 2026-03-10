import { useRef } from 'react'
import s from './AttachButton.module.css'

export default function AttachButton({ onAttach, attachedFile, disabled }) {
  const fileRef = useRef(null)
  const hasFile = !!attachedFile

  function handleChange(e) {
    const file = e.target.files[0]
    if (file) { onAttach(file); e.target.value = '' }
  }

  return (
    <>
      <button
        className={`${s.seal} ${hasFile ? s.done : ''}`}
        onClick={() => fileRef.current.click()}
        disabled={disabled}
        title={hasFile ? attachedFile.name : 'Attach PDF'}
        aria-label="Attach PDF"
      >
        <svg className={s.ring} viewBox="0 0 36 36">
          <circle cx="18" cy="18" r="16" />
        </svg>
        <span className={s.icon}>{hasFile ? '✓' : '+'}</span>
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={handleChange}
      />
    </>
  )
}
