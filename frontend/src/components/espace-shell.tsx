"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

import { useAuth } from "@/components/auth-provider";
import { Logo, Spinner } from "@/components/ui";

const NAV = [
  { href: "/chat", label: "Questions" },
  { href: "/analyses/nouvelle", label: "Analyser un projet" },
  { href: "/analyses", label: "Mes analyses" },
];

/** Espace connecté (rendu dans le navigateur : la session y vit). Sans session : connexion. */
export function EspaceShell({ children }: { children: React.ReactNode }) {
  const { session, loading, anonymous, email, signOut } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !session) router.replace("/connexion");
  }, [loading, session, router]);

  if (loading || !session) {
    return (
      <div className="flex flex-1 items-center justify-center text-slate-500">
        <Spinner />
      </div>
    );
  }

  const active = (href: string) =>
    href === "/analyses"
      ? pathname === "/analyses" || (pathname.startsWith("/analyses/") && pathname !== "/analyses/nouvelle")
      : pathname.startsWith(href);

  return (
    <div className="flex h-dvh flex-col">
      <header className="shrink-0 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link href="/chat">
            <Logo />
          </Link>
          {/* petit écran : le menu passe sur une seconde ligne */}
          <nav className="order-last -mx-1 flex w-full gap-1 overflow-x-auto text-sm md:order-none md:mx-0 md:w-auto">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`whitespace-nowrap rounded-md px-3 py-1.5 font-medium ${
                  active(item.href) ? "bg-indigo-50 text-indigo-800" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="hidden text-slate-500 md:inline">
              {anonymous ? "Session d'essai" : email}
            </span>
            <button
              onClick={async () => {
                // une session d'essai ne se retrouve pas : ses analyses seraient perdues
                const warning = "Session d'essai : vos analyses et conversations seront perdues. Se déconnecter ?";
                if (anonymous && !window.confirm(warning)) return;
                await signOut();
                router.replace("/");
              }}
              className="whitespace-nowrap rounded-md px-2 py-1 text-slate-600 hover:bg-slate-100"
            >
              Se déconnecter
            </button>
          </div>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
    </div>
  );
}
