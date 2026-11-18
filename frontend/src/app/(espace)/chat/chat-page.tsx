"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button, ErrorBox, Prose, Spinner } from "@/components/ui";
import { ApiError, api, json, stream } from "@/lib/api";
import type { ChatEvent, ConversationSummary, Source, StoredMessage, Verification } from "@/lib/types";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  verification?: Verification | null;
  refused?: boolean;
  step?: "recherche" | "redaction" | null; // réponse en cours
};

const EXAMPLES = [
  "Quelle amende pour un système de notation sociale ?",
  "Quand les obligations des systèmes à haut risque s'appliquent-elles ?",
  "Un chatbot doit-il dire qu'il est une IA ?",
];

/** [3] dans une réponse -> lien vers la source 3 (sauf s'il s'agit déjà d'un lien Markdown). */
function linkCitations(text: string): string {
  return text.replace(/\[(\d{1,2})\](?!\()/g, "[\\[$1\\]](#source-$1)");
}

export function ChatPage() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selected, setSelected] = useState<number | null>(null); // message dont on voit les sources
  const [highlight, setHighlight] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const loadConversations = useCallback(async () => {
    try {
      setConversations(await api<ConversationSummary[]>("/conversations"));
    } catch (e) {
      setError(e as ApiError);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- chargement initial depuis l'API
    loadConversations();
  }, [loadConversations]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function open(id: string) {
    setError(null);
    setCurrent(id);
    setSelected(null);
    try {
      const data = await api<{ messages: StoredMessage[] }>(`/conversations/${id}`);
      const loaded = data.messages.map((m) => ({ role: m.role, content: m.content, sources: m.sources ?? [] }));
      setMessages(loaded);
      setSelected(loaded.length ? loaded.length - 1 : null);
    } catch (e) {
      setError(e as ApiError);
    }
  }

  function startNew() {
    setCurrent(null);
    setMessages([]);
    setSelected(null);
    setError(null);
  }

  async function remove(id: string) {
    if (!window.confirm("Effacer cette conversation et ses messages ?")) return;
    try {
      await api(`/conversations/${id}`, { method: "DELETE" });
      if (current === id) startNew();
      await loadConversations();
    } catch (e) {
      setError(e as ApiError);
    }
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setError(null);
    setInput("");
    setBusy(true);
    const index = messages.length + 1; // position de la réponse
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", step: "recherche" }]);
    setSelected(index);
    const update = (patch: Partial<Message> | ((m: Message) => Partial<Message>)) =>
      setMessages((all) =>
        all.map((m, i) => (i === index ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)),
      );
    const isNew = current === null;
    try {
      await stream<ChatEvent>("/chat/stream", json({ message: question, session_id: current }), (event) => {
        if (event.type === "statut") update({ step: event.etape });
        else if (event.type === "sources") {
          setCurrent(event.session_id);
          update({ sources: event.sources });
        } else if (event.type === "texte") update((m) => ({ content: m.content + event.texte }));
        else if (event.type === "fin") {
          const cited = new Set(event.verification?.cited ?? []);
          update((m) => ({
            content: event.answer, // texte définitif (refus, avertissement sur un chiffre)
            refused: event.refused,
            verification: event.verification,
            step: null,
            sources: (m.sources ?? []).map((s) => ({ ...s, cited: cited.has(s.n) })),
          }));
        }
      });
      if (isNew) await loadConversations();
    } catch (e) {
      setMessages((all) => all.slice(0, index - 1)); // la question n'a pas été traitée
      setInput(question);
      setError(e as ApiError);
    } finally {
      setBusy(false);
    }
  }

  function showSource(messageIndex: number, n: number) {
    setSelected(messageIndex);
    setHighlight(n);
    requestAnimationFrame(() =>
      document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }

  const panel = selected !== null ? messages[selected] : null;

  return (
    <div className="mx-auto flex h-full max-w-7xl">
      {/* conversations */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white md:flex">
        <div className="p-3">
          <Button variant="secondary" className="w-full" onClick={startNew}>
            Nouvelle conversation
          </Button>
        </div>
        <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-3 text-sm">
          {conversations.map((c) => (
            <li key={c.id} className="group flex items-center">
              <button
                onClick={() => open(c.id)}
                className={`min-w-0 flex-1 truncate rounded-md px-3 py-2 text-left ${
                  current === c.id ? "bg-indigo-50 font-medium text-indigo-800" : "text-slate-700 hover:bg-slate-50"
                }`}
                title={c.titre}
              >
                {c.titre}
              </button>
              <button
                onClick={() => remove(c.id)}
                aria-label="Effacer la conversation"
                className="ml-1 hidden rounded px-1.5 text-slate-400 hover:bg-red-50 hover:text-red-700 group-hover:block"
              >
                ×
              </button>
            </li>
          ))}
          {!conversations.length && <li className="px-3 py-2 text-slate-400">Aucune conversation</li>}
        </ul>
      </aside>

      {/* échanges */}
      <section className="flex min-w-0 flex-1 flex-col">
        {/* petit écran : la liste des conversations devient un menu */}
        <div className="flex gap-2 border-b border-slate-200 bg-white px-4 py-2 md:hidden">
          <select
            value={current ?? ""}
            onChange={(e) => (e.target.value ? open(e.target.value) : startNew())}
            className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm"
            aria-label="Conversation"
          >
            <option value="">Nouvelle conversation</option>
            {conversations.map((c) => (
              <option key={c.id} value={c.id}>
                {c.titre}
              </option>
            ))}
          </select>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
          {!messages.length && (
            <div className="mx-auto mt-10 max-w-xl text-center">
              <h1 className="text-2xl font-semibold text-slate-900">Posez votre question</h1>
              <p className="mt-2 text-sm text-slate-600">
                Réponses tirées de l&apos;AI Act, du RGPD et des fiches de la CNIL, avec les articles
                cités. Sans source suffisante, ConformIA le dit plutôt que d&apos;inventer.
              </p>
              <div className="mt-6 flex flex-col gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-left text-sm text-slate-700 hover:border-indigo-300 hover:bg-indigo-50/40"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="mx-auto max-w-3xl space-y-5">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-indigo-700 px-4 py-2.5 text-sm text-white">
                    {m.content}
                  </p>
                </div>
              ) : (
                <div
                  key={i}
                  className={`rounded-2xl border bg-white px-5 py-4 text-sm ${
                    selected === i ? "border-indigo-200 ring-1 ring-indigo-100" : "border-slate-200"
                  }`}
                  onClick={() => setSelected(i)}
                >
                  {m.step && !m.content && (
                    <p className="flex items-center gap-2 text-slate-500">
                      <Spinner />
                      {m.step === "recherche" ? "Recherche dans les textes…" : "Rédaction de la réponse…"}
                    </p>
                  )}
                  {m.content && <Prose onSource={(n) => showSource(i, n)}>{linkCitations(m.content)}</Prose>}
                  {!!m.verification?.unverified_quotes.length && (
                    <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
                      Un extrait entre guillemets n&apos;a pas été retrouvé mot pour mot dans les sources :
                      vérifiez-le dans le texte officiel.
                    </p>
                  )}
                  {!!m.sources?.length && !m.refused && (
                    // petit écran : pas de panneau latéral, les sources se déplient sous la réponse
                    <details className="mt-3 lg:hidden">
                      <summary className="cursor-pointer text-xs font-medium text-indigo-700">
                        {m.sources.filter((s) => s.cited).length} sources citées sur {m.sources.length}
                      </summary>
                      <ul className="mt-2 space-y-1 text-xs">
                        {m.sources.map((s) => (
                          <li key={s.n} className={s.cited ? "" : "opacity-60"}>
                            <a href={s.url} target="_blank" rel="noreferrer" className="text-slate-800 underline">
                              [{s.n}] {s.label}
                            </a>{" "}
                            <span className="text-slate-500">· {s.document}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              ),
            )}
            <div ref={bottom} />
          </div>
        </div>
        <div className="shrink-0 border-t border-slate-200 bg-white px-4 py-3 sm:px-8">
          <form
            className="mx-auto flex max-w-3xl flex-col gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            {error && <ErrorBox message={error.message} retryAt={error.retryAt} />}
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                rows={1}
                maxLength={2000}
                placeholder="Votre question…"
                className="min-h-10 flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
              />
              <Button type="submit" disabled={busy || !input.trim()}>
                {busy ? <Spinner /> : "Envoyer"}
              </Button>
            </div>
          </form>
        </div>
      </section>

      {/* sources de la réponse choisie */}
      <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-4 lg:block">
        <h2 className="text-sm font-semibold text-slate-900">Sources</h2>
        {!panel?.sources?.length ? (
          <p className="mt-2 text-sm text-slate-500">Les textes utilisés pour une réponse s&apos;affichent ici.</p>
        ) : (
          <ol className="mt-3 space-y-2">
            {panel.sources.map((s) => (
              <li
                key={s.n}
                id={`source-${s.n}`}
                className={`rounded-lg border p-3 text-sm transition-colors ${
                  highlight === s.n ? "border-indigo-300 bg-indigo-50" : "border-slate-200"
                } ${s.cited ? "" : "opacity-70"}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <a href={s.url} target="_blank" rel="noreferrer" className="font-medium text-slate-900 hover:text-indigo-700">
                    [{s.n}] {s.label}
                  </a>
                  {s.cited && (
                    <span className="shrink-0 rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] font-medium text-emerald-700">
                      citée
                    </span>
                  )}
                </div>
                <p className="mt-1 text-xs text-slate-500">{s.document}</p>
                {s.cited_by && <p className="mt-1 text-xs text-slate-400">jointe car citée par {s.cited_by}</p>}
              </li>
            ))}
          </ol>
        )}
      </aside>
    </div>
  );
}
