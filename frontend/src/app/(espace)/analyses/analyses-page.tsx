"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ErrorBox, LevelBadge, Spinner, StatusBadge } from "@/components/ui";
import { ApiError, api, formatDate } from "@/lib/api";
import type { AnalysisSummary } from "@/lib/types";

export function MyAnalysesPage() {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const load = useCallback(async () => {
    try {
      setItems(await api<AnalysisSummary[]>("/analyses"));
    } catch (e) {
      setError(e as ApiError);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- chargement initial depuis l'API
    load();
  }, [load]);

  async function remove(item: AnalysisSummary) {
    if (!window.confirm(`Effacer « ${item.titre} » et son rapport ? C'est définitif.`)) return;
    setError(null);
    try {
      await api(`/analyses/${item.id}`, { method: "DELETE" });
      await load();
    } catch (e) {
      setError(e as ApiError);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold text-slate-900">Mes analyses</h1>
        <Link
          href="/analyses/nouvelle"
          className="rounded-lg bg-indigo-700 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-800"
        >
          Nouvelle analyse
        </Link>
      </div>
      {error && (
        <div className="mt-4">
          <ErrorBox message={error.message} />
        </div>
      )}
      {items === null && !error && (
        <p className="mt-8 flex items-center gap-2 text-sm text-slate-500">
          <Spinner /> Chargement…
        </p>
      )}
      {items?.length === 0 && (
        <p className="mt-8 rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
          Aucune analyse pour l&apos;instant.
        </p>
      )}
      {!!items?.length && (
        <ul className="mt-6 divide-y divide-slate-200 rounded-xl border border-slate-200 bg-white">
          {items.map((item) => (
            <li key={item.id} className="flex flex-wrap items-center gap-3 px-5 py-4">
              <Link href={`/analyses/${item.id}`} className="min-w-0 flex-1">
                <p className="truncate font-medium text-slate-900 hover:text-indigo-700">{item.titre}</p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {item.source === "document" ? "PDF" : "Description"} · {formatDate(item.created_at)}
                </p>
              </Link>
              {item.niveau && <LevelBadge level={item.niveau} />}
              <StatusBadge status={item.statut} />
              <button
                onClick={() => remove(item)}
                className="rounded-md px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-700"
              >
                Effacer
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-6 text-xs text-slate-500">
        Le nombre d&apos;analyses par jour est limité, pour que tout le monde puisse essayer.
      </p>
    </div>
  );
}
