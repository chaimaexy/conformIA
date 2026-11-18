"use client";

import type { Session } from "@supabase/supabase-js";
import { createContext, useContext, useEffect, useState } from "react";

import { supabase } from "@/lib/supabase";

type Auth = {
  session: Session | null;
  loading: boolean; // session pas encore lue (premier rendu)
  anonymous: boolean;
  email: string | null;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<Auth | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // la session vit dans le navigateur : lue après le premier rendu, puis suivie
    const client = supabase();
    client.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data } = client.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => data.subscription.unsubscribe();
  }, []);

  const user = session?.user;
  const value: Auth = {
    session,
    loading,
    anonymous: Boolean(user?.is_anonymous),
    email: user?.email || null,
    signOut: async () => {
      await supabase().auth.signOut();
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): Auth {
  const auth = useContext(AuthContext);
  if (!auth) throw new Error("useAuth hors de AuthProvider");
  return auth;
}

/** Messages de Supabase Auth, en français. */
export function authMessage(code: string | undefined, fallback: string): string {
  const messages: Record<string, string> = {
    invalid_credentials: "E-mail ou mot de passe incorrect.",
    email_not_confirmed: "Adresse pas encore confirmée : ouvrez le lien reçu par e-mail.",
    user_already_exists: "Un compte existe déjà avec cette adresse.",
    weak_password: "Mot de passe trop faible : au moins 8 caractères.",
    email_address_not_authorized:
      "Cette adresse ne peut pas recevoir l'e-mail de confirmation pour l'instant. Utilisez « Essayer sans compte ».",
    over_email_send_rate_limit: "Trop d'e-mails envoyés : réessayez dans une heure.",
    over_request_rate_limit: "Trop de tentatives : réessayez dans quelques minutes.",
    anonymous_provider_disabled: "L'essai sans compte est désactivé pour le moment.",
  };
  return (code && messages[code]) || fallback;
}
