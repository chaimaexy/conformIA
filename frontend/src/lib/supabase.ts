import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Clé publiable : faite pour le navigateur (les données sont protégées par RLS et par l'API).
// La clé secrète du projet ne va jamais ici.
let client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (!client) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
    if (!url || !key) {
      throw new Error(
        "NEXT_PUBLIC_SUPABASE_URL et NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY manquent (frontend/.env.local)",
      );
    }
    // session gardée dans le navigateur (localStorage) et renouvelée automatiquement
    client = createClient(url, key, { auth: { persistSession: true, autoRefreshToken: true } });
  }
  return client;
}
