/** Card with a cursor-following spotlight border (shared by the explainer pages). */
export function Spotlight({ className = '', children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const onMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`);
  };
  return <button type="button" className={`ix-spot ${className}`} onPointerMove={onMove} {...rest}>{children}</button>;
}
