"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, ErrorBox, LEVELS, LevelBadge, Prose, Spinner } from "@/components/ui";
import { ApiError, api, formatDate, json, stream } from "@/lib/api";
import type { AnalysisEvent, AnalysisState, Question, Report, Ticket } from "@/lib/types";

const STEPS: [string, string][] = [
  ["profil", "Profil du projet"],
  ["questions", "Vos réponses"],
  ["interdictions", "Pratiques interdites (article 5)"],
  ["haut_risque", "Haut risque (article 6 et annexe III)"],
  ["obligations", "Obligations et dates"],
  ["rgpd", "Liste de vérification de la CNIL"],
  ["rapport", "Rapport"],
];

type Live = {
  position: number | null; // place dans la file, tant que l'analyse attend son tour
  done: string[]; // étapes terminées
  questions: Question[] | null;
  report: { rapport: Report; markdown: string } | null;
  failure: string | null;
};

function stepsThrough(step: string, done: string[]): string[] {
  const index = STEPS.findIndex(([key]) => key === step);
  if (index === -1) return done;
  return Array.from(new Set([...done, ...STEPS.slice(0, index + 1).map(([key]) => key)]));
}

const EMPTY: Live ={ position: null, done: [], questions: null, report: null, failure: null };

export function AnalysisView({ id }: { id: string }) {
  const [title, setTitle] = useState<string>("");
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  const [live, setLive] = useState<Live>(EMPTY);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const abort = useRef<AbortController | null>(null);

  const follow = useCallback(async (flux: string) => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setLive((l) => ({ ...l, questions: null, failure: null }));
    try {
      await stream<AnalysisEvent>(
        flux,
        {},
        (event) =>
          setLive((l) => {
            switch (event.type) {
              case "en_attente":
                return { ...l, position: event.position };
              case "debut":
                return { ...l, position: null };
              case "etape":
                // les étapes se suivent : celle-ci finie, les précédentes le sont aussi (utile
                // quand le flux reprend après les questions, page rechargée)
                return { ...l, position: null, done: stepsThrough(event.etape, l.done) };
              case "questions":
                return { ...l, position: null, questions: event.questions };
              case "rapport":
                return { ...l, position: null, report: { rapport: event.rapport, markdown: event.markdown } };
              case "erreur":
                return { ...l, position: null, failure: event.message };
            }
          }),
        controller.signal,
      );
    } catch (e) {
      setError(e as ApiError);
    } finally {
      if (abort.current === controller) setRunning(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api<AnalysisState & { flux: string }>(`/analyses/${id}`)
      .then((state) => {
        if (cancelled) return;
        setTitle(state.titre);
        setCreatedAt(state.cree_le);
        if (state.statut === "termine" && state.rapport) {
          setLive({ ...EMPTY, done: STEPS.map(([k]) => k), report: { rapport: state.rapport, markdown: state.markdown ?? "" } });
        } else if (state.statut === "questions") {
          setLive({ ...EMPTY, done: ["profil"], questions: state.questions });
        } else if (state.statut === "erreur") {
          setLive({ ...EMPTY, failure: "L'analyse a échoué. Relancez-la dans quelques minutes." });
        } else {
          follow(state.flux); // en attente ou en cours : on suit le flux
        }
      })
      .catch((e) => !cancelled && setError(e as ApiError));
    return () => {
      cancelled = true;
      abort.current?.abort();
    };
  }, [id, follow]);

  async function answer(answers: Record<string, string | string[]>) {
    setError(null);
    try {
      const ticket = await api<Ticket>(`/analyses/${id}/reponses`, json({ reponses: answers }));
      await follow(ticket.flux);
    } catch (e) {
      setError(e as ApiError);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <Link href="/analyses" className="text-sm text-slate-500 hover:text-slate-800">
        ← Mes analyses
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-slate-900">{title || "Analyse"}</h1>
      {createdAt && <p className="mt-1 text-sm text-slate-500">Lancée le {formatDate(createdAt)}</p>}

      <div className="mt-6 grid gap-6 lg:grid-cols-[16rem_1fr]">
        <Steps live={live} running={running} />
        <div className="min-w-0 space-y-4">
          {error && <ErrorBox message={error.message} retryAt={error.retryAt} />}
          {live.failure && <ErrorBox message={live.failure} />}
          {live.position !== null && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600">
              <p className="flex items-center gap-2 font-medium text-slate-900">
                <Spinner /> {live.position <= 1 ? "Votre analyse est la prochaine." : `Position dans la file : ${live.position}.`}
              </p>
              <p className="mt-2">
                Les analyses passent une à une (l&apos;offre gratuite de Mistral limite les requêtes). Une
                analyse prend 2 à 4 minutes : vous pouvez quitter cette page et revenir.
              </p>
            </div>
          )}
          {running && live.position === null && !live.questions && !live.report && !live.failure && (
            <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600">
              <Spinner /> Analyse en cours…
            </div>
          )}
          {live.questions && <QuestionsForm questions={live.questions} onSubmit={answer} />}
          {live.report && <ReportView report={live.report.rapport} markdown={live.report.markdown} title={title} />}
        </div>
      </div>
    </div>
  );
}

function Steps({ live, running }: { live: Live; running: boolean }) {
  const next = STEPS.find(([k]) => !live.done.includes(k))?.[0];
  return (
    <ol className="h-fit space-y-1 rounded-xl border border-slate-200 bg-white p-4 text-sm">
      {STEPS.map(([key, label]) => {
        const done = live.done.includes(key);
        const current = running && key === next && live.position === null && !live.questions;
        const waiting = key === "questions" && !!live.questions;
        return (
          <li key={key} className="flex items-center gap-2.5 py-1">
            <span
              className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[11px] font-bold ${
                done
                  ? "bg-emerald-600 text-white"
                  : current || waiting
                    ? "bg-indigo-100 text-indigo-700"
                    : "bg-slate-100 text-slate-400"
              }`}
            >
              {done ? "✓" : current ? <Spinner className="h-3 w-3" /> : waiting ? "?" : ""}
            </span>
            <span className={done || current || waiting ? "text-slate-900" : "text-slate-400"}>{label}</span>
          </li>
        );
      })}
    </ol>
  );
}

function QuestionsForm({
  questions,
  onSubmit,
}: {
  questions: Question[];
  onSubmit: (answers: Record<string, string | string[]>) => Promise<void>;
}) {
  const [answers, setAnswers] = useState<Record<string, string | string[]>>({});
  const [busy, setBusy] = useState(false);
  const complete = questions.every((q) => {
    const a = answers[q.champ];
    return Array.isArray(a) ? a.length > 0 : Boolean(a);
  });

  function toggle(q: Question, value: string) {
    setAnswers((all) => {
      if (!q.multiple) return { ...all, [q.champ]: value };
      const current = (all[q.champ] as string[] | undefined) ?? [];
      return { ...all, [q.champ]: current.includes(value) ? current.filter((v) => v !== value) : [...current, value] };
    });
  }

  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        await onSubmit(answers);
        setBusy(false);
      }}
      className="rounded-xl border border-amber-200 bg-white p-5"
    >
      <h2 className="font-semibold text-slate-900">Quelques questions pour conclure</h2>
      <p className="mt-1 text-sm text-slate-600">
        Ces points ne ressortent pas clairement de votre projet. « Je ne sais pas » est une réponse
        acceptée : le rapport dira ce qui reste à vérifier.
      </p>
      <div className="mt-5 space-y-5">
        {questions.map((q) => (
          <fieldset key={q.champ}>
            <legend className="text-sm font-medium text-slate-900">{q.question}</legend>
            {q.aide && <p className="mt-0.5 text-xs text-slate-500">{q.aide}</p>}
            <div className="mt-2 flex flex-wrap gap-2">
              {q.options.map((o) => {
                const value = answers[q.champ];
                const checked = Array.isArray(value) ? value.includes(o.valeur) : value === o.valeur;
                return (
                  <label
                    key={o.valeur}
                    className={`cursor-pointer rounded-lg border px-3 py-1.5 text-sm ${
                      checked ? "border-indigo-500 bg-indigo-50 text-indigo-800" : "border-slate-300 text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <input
                      type={q.multiple ? "checkbox" : "radio"}
                      name={q.champ}
                      value={o.valeur}
                      checked={checked}
                      onChange={() => toggle(q, o.valeur)}
                      className="sr-only"
                    />
                    {o.libelle}
                  </label>
                );
              })}
            </div>
          </fieldset>
        ))}
      </div>
      <Button type="submit" disabled={!complete || busy} className="mt-6">
        {busy && <Spinner />} Envoyer mes réponses
      </Button>
    </form>
  );
}

function ReportView({ report, markdown, title }: { report: Report; markdown: string; title: string }) {
  const level = report.classification.niveau;

  function download() {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `rapport-conformia-${title.replace(/[^\p{L}\p{N}]+/gu, "-").slice(0, 40) || "analyse"}.md`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <LevelBadge level={level} />
            <span className="text-sm font-medium text-slate-900">{LEVELS[level].label}</span>
          </div>
          <Button variant="secondary" onClick={download}>
            Télécharger le rapport (.md)
          </Button>
        </div>
      </div>
      <article className="rounded-xl border border-slate-200 bg-white p-6 text-sm">
        <Prose>{markdown}</Prose>
      </article>
    </div>
  );
}
