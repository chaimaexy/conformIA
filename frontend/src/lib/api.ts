// Appels à l'API FastAPI, avec le jeton de la session Supabase.
//
// Les flux (chat, analyses) sont des Server-Sent Events lus avec fetch : EventSource ne sait pas
// envoyer l'en-tête Authorization.

import { supabase } from "./supabase";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public retryAt: string | null = null, // quota : quand une place se libère
  ) {
    super(message);
  }
}

async function authHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase().auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function toError(response: Response): Promise<ApiError> {
  let detail: unknown = null;
  try {
    detail = (await response.json()).detail;
  } catch {
    // réponse sans JSON
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    const d = detail as { message: string; prochaine?: string };
    return new ApiError(response.status, d.message, d.prochaine ?? null);
  }
  if (typeof detail === "string") return new ApiError(response.status, detail);
  if (Array.isArray(detail)) return new ApiError(response.status, "Requête invalide.");
  if (response.status === 401) return new ApiError(401, "Session expirée : reconnectez-vous.");
  return new ApiError(response.status, "Le service ne répond pas. Réessayez dans un instant.");
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_URL + path, {
      ...init,
      headers: { ...(init.headers ?? {}), ...(await authHeaders()) },
    });
  } catch {
    throw new ApiError(0, "Le serveur de ConformIA est injoignable.");
  }
  if (!response.ok) throw await toError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function json(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

/** Lit un flux SSE (`data: {...}` séparés par une ligne vide) et appelle onEvent pour chacun. */
export async function stream<E>(
  path: string,
  init: RequestInit,
  onEvent: (event: E) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(API_URL + path, {
      ...init,
      signal,
      headers: { ...(init.headers ?? {}), ...(await authHeaders()) },
    });
  } catch (error) {
    if (signal?.aborted) return;
    throw error instanceof ApiError ? error : new ApiError(0, "Le serveur de ConformIA est injoignable.");
  }
  if (!response.ok || !response.body) throw await toError(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let end;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        for (const line of block.split("\n")) {
          if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as E);
        }
      }
    }
  } catch (error) {
    if (!signal?.aborted) throw error;
  }
}

/** Heure locale de l'utilisateur pour une date ISO (« 14:05 », ou « 18 nov., 14:05 »). */
export function formatDate(iso: string, withDay = true): string {
  const date = new Date(iso);
  return date.toLocaleString("fr-FR", {
    ...(withDay ? { day: "numeric", month: "short" } : {}),
    hour: "2-digit",
    minute: "2-digit",
  });
}
