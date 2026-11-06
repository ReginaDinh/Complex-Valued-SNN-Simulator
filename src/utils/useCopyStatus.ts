import { useCallback, useEffect, useRef, useState } from 'react';
import { copyText } from './clipboard';

export type CopyStatus = 'idle' | 'copied' | 'failed';

// Copy-to-clipboard with visible feedback: 'copied' or 'failed', auto-reset to 'idle'.
export function useCopyStatus(resetMs: number = 2000) {
  const [status, setStatus] = useState<CopyStatus>('idle');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    },
    []
  );

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      const ok = await copyText(text);
      setStatus(ok ? 'copied' : 'failed');
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      // Failures stay visible a bit longer so the user can read the hint
      timerRef.current = setTimeout(() => setStatus('idle'), ok ? resetMs : resetMs + 2000);
      return ok;
    },
    [resetMs]
  );

  return { status, copy };
}
