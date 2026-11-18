"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button, ErrorBox, Spinner } from "@/components/ui";
import { ApiError, api, json } from "@/lib/api";
import type { Ticket } from "@/lib/types";

const MAX_BYTES = 10 * 1024 * 1024; // mêmes limites que l'API (backend/app/agents/document.py)
const EXAMPLE =
  "Nous sommes une PME de 80 salariés. Nous voulons utiliser un logiciel acheté à un éditeur qui " +
  "analyse automatiquement les CV reçus et classe les candidats pour nos recrutements.";

export function NewAnalysisPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"document" | "description">("document");
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fileProblem =
    file && file.size > MAX_BYTES ? "Ce fichier dépasse 10 Mo." : file && !/\.pdf$/i.test(file.name) ? "Choisissez un fichier PDF." : null;
  const ready = mode === "document" ? Boolean(file) && !fileProblem : text.trim().length >= 20;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      let ticket: Ticket;
      if (mode === "document" && file) {
        const form = new FormData();
        form.append("fichier", file);
        ticket = await api<Ticket>("/analyses/document", { method: "POST", body: form });
      } else {
        ticket = await api<Ticket>("/analyses", json({ description: text.trim() }));
      }
      router.push(`/analyses/${ticket.analyse_id}`);
    } catch (e) {
      setError(e as ApiError);
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <h1 className="text-2xl font-semibold text-slate-900">Analyser un projet</h1>
      <p className="mt-2 text-sm text-slate-600">
        L&apos;agent qualifie votre système au regard de l&apos;AI Act, vous pose les questions qui
        manquent, puis liste les obligations, leurs dates et les points RGPD de la CNIL. Comptez 2 à
        4 minutes.
      </p>

      <form onSubmit={submit} className="mt-6 rounded-xl border border-slate-200 bg-white p-6">
        <div className="mb-5 inline-grid grid-cols-2 rounded-lg bg-slate-100 p-1 text-sm">
          {(
            [
              ["document", "Envoyer un PDF"],
              ["description", "Décrire le projet"],
            ] as const
          ).map(([m, label]) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-md px-4 py-1.5 font-medium ${mode === m ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"}`}
            >
              {label}
            </button>
          ))}
        </div>

        {mode === "document" ? (
          <div>
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 px-6 py-10 text-center hover:border-indigo-400 hover:bg-indigo-50/30">
              <input
                type="file"
                accept="application/pdf,.pdf"
                className="sr-only"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span className="text-sm font-medium text-slate-900">
                {file ? file.name : "Choisir un PDF : cahier des charges, documentation, AIPD…"}
              </span>
              <span className="mt-1 text-xs text-slate-500">
                {file ? `${(file.size / 1024 / 1024).toFixed(1)} Mo` : "PDF avec du texte (pas un scan), 30 pages et 10 Mo au plus"}
              </span>
            </label>
            {fileProblem && <p className="mt-2 text-sm text-red-700">{fileProblem}</p>}
            <p className="mt-3 text-xs text-slate-500">
              Le PDF est lu tout de suite, en mémoire, puis oublié : il n&apos;est pas conservé.
              Chaque élément du profil sera prouvé par un extrait du document et sa page.
            </p>
          </div>
        ) : (
          <div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
              maxLength={5000}
              placeholder="Que fait le système, qui l'utilise, sur quelles données, qui l'a développé ?"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
            <div className="mt-1 flex justify-between text-xs text-slate-500">
              <button type="button" onClick={() => setText(EXAMPLE)} className="text-indigo-700 hover:underline">
                Utiliser un exemple
              </button>
              <span>{text.length} / 5000</span>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4">
            <ErrorBox message={error.message} retryAt={error.retryAt} />
          </div>
        )}
        <Button type="submit" disabled={!ready || busy} className="mt-5">
          {busy && <Spinner />} Lancer l&apos;analyse
        </Button>
      </form>
    </div>
  );
}
