import Link from "next/link";

import { TryButton } from "@/components/try-button";
import { Logo } from "@/components/ui";

const FEATURES = [
  {
    title: "Questions sourcées",
    text: "Posez une question sur l'AI Act, le RGPD ou les fiches de la CNIL. Chaque réponse cite les articles utilisés, avec un lien vers le texte officiel. Les citations et les chiffres sont vérifiés par le code ; sans source suffisante, ConformIA préfère ne pas répondre.",
  },
  {
    title: "Pré-diagnostic d'un projet",
    text: "Décrivez votre projet ou envoyez son cahier des charges en PDF. L'agent qualifie le système au regard de l'AI Act, pose les questions qui manquent, puis liste les obligations, leurs dates et les points RGPD de la CNIL.",
  },
  {
    title: "Vos données restent les vôtres",
    text: "Le PDF est lu puis oublié. Vos analyses et conversations ne sont visibles que par vous, et vous pouvez les effacer à tout moment.",
  },
];

export default function Home() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-4 py-5 sm:px-6">
        <Logo className="text-lg" />
        <Link href="/connexion" className="text-sm font-medium text-slate-700 hover:text-indigo-700">
          Se connecter
        </Link>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6">
        <section className="py-14 sm:py-20">
          <p className="text-sm font-medium uppercase tracking-wide text-indigo-700">
            AI Act · RGPD · CNIL
          </p>
          <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
            La conformité de vos projets d&apos;IA, réponse sourcée à l&apos;appui.
          </h1>
          <p className="mt-5 max-w-2xl text-lg text-slate-600">
            ConformIA s&apos;appuie sur les textes officiels : le règlement européen sur l&apos;IA
            dans sa version consolidée, le RGPD et les fiches pratiques de la CNIL.
          </p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-start">
            <TryButton />
            <Link
              href="/connexion"
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50"
            >
              Créer un compte
            </Link>
          </div>
        </section>

        <section className="grid gap-4 pb-16 sm:grid-cols-3">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-xl border border-slate-200 bg-white p-5">
              <h2 className="font-semibold text-slate-900">{f.title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{f.text}</p>
            </div>
          ))}
        </section>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <p className="mx-auto max-w-5xl px-4 py-5 text-xs text-slate-500 sm:px-6">
          Pré-diagnostic automatique : il ne remplace pas l&apos;avis d&apos;un juriste ou de votre
          délégué à la protection des données. Fiches CNIL : licence CC-BY-ND 4.0, source CNIL.
        </p>
      </footer>
    </div>
  );
}
