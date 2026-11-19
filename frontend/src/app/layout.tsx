import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { AuthProvider } from "@/components/auth-provider";
import { ServerWake } from "@/components/server-wake";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ConformIA — AI Act, RGPD, CNIL",
  description:
    "Copilote de conformité IA : réponses sourcées sur l'AI Act, le RGPD et les fiches de la CNIL, et pré-diagnostic d'un projet.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fr" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <AuthProvider>{children}</AuthProvider>
        <ServerWake />
      </body>
    </html>
  );
}
