"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { authMessage } from "@/components/auth-provider";
import { TryButton } from "@/components/try-button";
import { Button, ErrorBox, Logo, Spinner } from "@/components/ui";
import { supabase } from "@/lib/supabase";

export default function Connexion() {
  const router = useRouter();
  const [mode, setMode] = useState<"connexion" | "inscription">("connexion");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    const auth = supabase().auth;
    if (mode === "connexion") {
      const { error: failure } = await auth.signInWithPassword({ email, password });
      setBusy(false);
      if (failure) return setError(authMessage(failure.code, "Connexion impossible."));
      router.push("/chat");
      return;
    }
    const { data, error: failure } = await auth.signUp({ email, password });
    setBusy(false);
    if (failure) return setError(authMessage(failure.code, "Inscription impossible."));
    if (data.session) router.push("/chat");
    else setNotice("Compte créé : ouvrez le lien de confirmation reçu par e-mail, puis connectez-vous.");
  }

  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <Link href="/" className="mb-8 flex justify-center">
          <Logo className="text-xl" />
        </Link>
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-5 grid grid-cols-2 rounded-lg bg-slate-100 p-1 text-sm">
            {(["connexion", "inscription"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`rounded-md py-1.5 font-medium ${mode === m ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"}`}
              >
                {m === "connexion" ? "Se connecter" : "Créer un compte"}
              </button>
            ))}
          </div>
          <form onSubmit={submit} className="space-y-4">
            <label className="block text-sm">
              <span className="font-medium text-slate-700">E-mail</span>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
              />
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">Mot de passe</span>
              <input
                type="password"
                required
                minLength={8}
                autoComplete={mode === "connexion" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
              />
            </label>
            {error && <ErrorBox message={error} />}
            {notice && (
              <p className="rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800">{notice}</p>
            )}
            <Button type="submit" disabled={busy} className="w-full">
              {busy && <Spinner />}
              {mode === "connexion" ? "Se connecter" : "Créer le compte"}
            </Button>
          </form>
        </div>
        <div className="mt-6 text-center text-sm text-slate-600">
          <p className="mb-3">Pas envie de créer un compte ?</p>
          <TryButton className="flex justify-center" />
          <p className="mt-3 text-xs text-slate-500">
            Session d&apos;essai : vos analyses sont perdues si vous vous déconnectez.
          </p>
        </div>
      </div>
    </main>
  );
}
