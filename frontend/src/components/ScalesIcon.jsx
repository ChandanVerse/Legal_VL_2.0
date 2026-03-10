export default function ScalesIcon({ size = 30, className }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 30 30"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <line x1="15" y1="3"  x2="15" y2="27" stroke="currentColor" strokeWidth="1.2" />
      <line x1="4"  y1="9"  x2="26" y2="9"  stroke="currentColor" strokeWidth="1.2" />
      <line x1="4"  y1="9"  x2="4"  y2="15" stroke="currentColor" strokeWidth=".9" />
      <line x1="26" y1="9"  x2="26" y2="15" stroke="currentColor" strokeWidth=".9" />
      <path d="M1 15 Q4 20 7 15"    stroke="currentColor" strokeWidth="1" fill="none" strokeLinecap="round" />
      <path d="M23 15 Q26 20 29 15" stroke="currentColor" strokeWidth="1" fill="none" strokeLinecap="round" />
      <line x1="11" y1="27" x2="19" y2="27" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  )
}
