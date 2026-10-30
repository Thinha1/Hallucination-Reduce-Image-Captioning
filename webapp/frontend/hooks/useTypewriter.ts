"use client";

import { useEffect, useRef, useState } from "react";

/** Reveals `text` a few characters at a time. Resets automatically whenever `text` changes
 * (e.g. a new reasoning paragraph comes back from a re-run of Step 4). */
export function useTypewriter(text: string, stepMs = 12, charsPerTick = 2): string {
  const [shown, setShown] = useState("");
  const prevText = useRef<string>("");

  useEffect(() => {
    if (text === prevText.current) return;
    prevText.current = text;
    setShown("");
    if (!text) return;

    let i = 0;
    const interval = setInterval(() => {
      i += charsPerTick;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, stepMs);
    return () => clearInterval(interval);
  }, [text, stepMs, charsPerTick]);

  return shown;
}
