import { ChatPage } from "./chat-page";

// Page de l'espace connecté : elle attend la session du navigateur, donc elle ne peut pas
// s'afficher instantanément (validation « instant » de Next.js 16, désactivée ici).
export const instant = false;

export default function Page() {
  return <ChatPage />;
}
