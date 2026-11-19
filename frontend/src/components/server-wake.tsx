"use client";

// Réveil de l'API à l'ouverture du site.
//
// En ligne, l'API tourne sur l'offre gratuite de Render : après 15 minutes sans visite, le
// serveur s'endort, et le réveil prend une à deux minutes. On l'appelle dès l'arrivée sur le site
// (pendant que l'on lit la page ou que l'on se connecte), et on prévient si la réponse tarde,
// pour qu'une première question lente ne passe pas pour une panne.

import { useEffect, useState } from "react";

import { API_URL } from "@/lib/api";
import { Spinner } from "./ui";

const SHOW_AFTER_MS = 3000; // en dessous, le serveur était déjà éveillé : rien à afficher
const RETRY_MS = 5000;
const GIVE_UP_MS = 240_000; // réveil chez Render (~1 min) + démarrage de l'API (~35 s)

type State = "checking" | "waking" | "ready" | "down";

export function ServerWake() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let stopped = false;
    const started = Date.now();
    const slow = setTimeout(() => !stopped && setState((s) => (s === "checking" ? "waking" : s)), SHOW_AFTER_MS);

    async function ping() {
      while (!stopped) {
        try {
          const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
          if (response.ok) {
            if (!stopped) setState("ready");
            return;
          }
        } catch {
          // serveur en cours de démarrage : on réessaie
        }
        if (Date.now() - started > GIVE_UP_MS) {
          if (!stopped) setState("down");
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, RETRY_MS));
      }
    }

    ping();
    return () => {
      stopped = true;
      clearTimeout(slow);
    };
  }, []);

  if (state !== "waking" && state !== "down") return null;
  return (
    <div
      role="status"
      className="fixed inset-x-4 bottom-4 z-50 mx-auto flex max-w-md items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 shadow-lg"
    >
      {state === "waking" ? (
        <>
          <Spinner className="mt-0.5 shrink-0" />
          <span>
            Le serveur de démonstration se réveille (hébergement gratuit, en veille après
            15 minutes sans visite). Comptez une à deux minutes.
          </span>
        </>
      ) : (
        <span>Le serveur de ConformIA ne répond pas. Réessayez dans quelques minutes.</span>
      )}
    </div>
  );
}
