import { Suspense } from "react";

import { EspaceShell } from "@/components/espace-shell";

// Tout l'espace connecté lit la session et l'adresse dans le navigateur : préparé à l'avance,
// il n'a que son indicateur de chargement (Next.js 16, Cache Components).
export default function EspaceLayout({ children }: LayoutProps<"/">) {
  return (
    <Suspense
      fallback={
        <div className="flex flex-1 items-center justify-center text-slate-500">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
        </div>
      }
    >
      <EspaceShell>{children}</EspaceShell>
    </Suspense>
  );
}
