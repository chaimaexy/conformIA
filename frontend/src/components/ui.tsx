"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AnalysisStatus, Level } from "@/lib/types";

export function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 font-semibold tracking-tight ${className}`}>
      <span
        aria-hidden
        className="grid h-7 w-7 place-items-center rounded-md bg-indigo-700 text-sm font-bold text-white"
      >
        C
      </span>
      <span>
        Conform<span className="text-indigo-700">IA</span>
      </span>
    </span>
  );
}

export function Button({
  variant = "primary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger" }) {
  const styles = {
    primary: "bg-indigo-700 text-white hover:bg-indigo-800 disabled:bg-indigo-300",
    secondary:
      "border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 disabled:text-slate-400",
    ghost: "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
    danger: "text-red-700 hover:bg-red-50",
  }[variant];
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed ${styles} ${className}`}
      {...props}
    />
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
    />
  );
}

export function ErrorBox({ message, retryAt }: { message: string; retryAt?: string | null }) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      {message}
      {retryAt && (
        <>
          {" "}
          Prochaine possible à{" "}
          {new Date(retryAt).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}.
        </>
      )}
    </div>
  );
}

export const LEVELS: Record<Level, { label: string; short: string; tone: string }> = {
  interdit: {
    label: "Pratique interdite (article 5)",
    short: "Interdit",
    tone: "bg-red-100 text-red-800 ring-red-200",
  },
  haut_risque: {
    label: "Système d'IA à haut risque (article 6, paragraphe 2, et annexe III)",
    short: "Haut risque",
    tone: "bg-orange-100 text-orange-800 ring-orange-200",
  },
  haut_risque_a_confirmer: {
    label: "Haut risque à confirmer (article 6, paragraphe 1, et annexe I)",
    short: "Haut risque à confirmer",
    tone: "bg-amber-100 text-amber-800 ring-amber-200",
  },
  incertain: {
    label: "Classification incertaine : des informations manquent pour conclure",
    short: "Incertain",
    tone: "bg-slate-100 text-slate-700 ring-slate-200",
  },
  obligations_specifiques: {
    label: "Obligations de transparence ou de modèle à usage général",
    short: "Obligations spécifiques",
    tone: "bg-sky-100 text-sky-800 ring-sky-200",
  },
  risque_minimal: {
    label: "Risque minimal : pas d'obligation spécifique au-delà de la maîtrise de l'IA",
    short: "Risque minimal",
    tone: "bg-emerald-100 text-emerald-800 ring-emerald-200",
  },
};

export function LevelBadge({ level }: { level: Level }) {
  const l = LEVELS[level];
  return (
    <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${l.tone}`}>
      {l.short}
    </span>
  );
}

export const STATUSES: Record<AnalysisStatus, string> = {
  en_attente: "Dans la file",
  en_cours: "En cours",
  questions: "Attend vos réponses",
  termine: "Terminée",
  erreur: "En échec",
};

export function StatusBadge({ status }: { status: AnalysisStatus }) {
  const tone = {
    en_attente: "bg-slate-100 text-slate-700",
    en_cours: "bg-indigo-50 text-indigo-700",
    questions: "bg-amber-50 text-amber-800",
    termine: "bg-emerald-50 text-emerald-700",
    erreur: "bg-red-50 text-red-700",
  }[status];
  return <span className={`inline-flex rounded-md px-2 py-0.5 text-xs font-medium ${tone}`}>{STATUSES[status]}</span>;
}

/** Markdown juridique. Les liens « #source-n » (renvois [n] d'une réponse) appellent onSource. */
export function Prose({ children, onSource }: { children: string; onSource?: (n: number) => void }) {
  return (
    <div className="prose-juridique">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: label }) => {
            const source = href?.match(/^#source-(\d+)$/);
            if (source && onSource) {
              return (
                <button
                  type="button"
                  onClick={() => onSource(Number(source[1]))}
                  className="mx-0.5 rounded bg-indigo-50 px-1 text-[0.8em] font-semibold text-indigo-700 hover:bg-indigo-100"
                >
                  {label}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {label}
              </a>
            );
          },
        }}
      >
        {children}
      </Markdown>
    </div>
  );
}
