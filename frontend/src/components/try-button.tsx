"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { authMessage, useAuth } from "@/components/auth-provider";
import { Button, Spinner } from "@/components/ui";
import { supabase } from "@/lib/supabase";

/** « Essayer sans compte » : session anonyme de Supabase, sans e-mail ni donnée personnelle. */
export function TryButton({ next = "/chat", className = "" }: { next?: string; className?: string }) {
  const router = useRouter();
  const { session } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setError(null);
    if (session) {
      router.push(next);
      return;
    }
    setBusy(true);
    const { error: failure } = await supabase().auth.signInAnonymously();
    setBusy(false);
    if (failure) {
      setError(authMessage(failure.code, "Impossible d'ouvrir une session d'essai. Réessayez."));
      return;
    }
    router.push(next);
  }

  return (
    <div className={className}>
      <Button onClick={start} disabled={busy} className="w-full sm:w-auto">
        {busy && <Spinner />}
        {session ? "Continuer" : "Essayer sans compte"}
      </Button>
      {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
    </div>
  );
}
