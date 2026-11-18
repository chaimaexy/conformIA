// Formats renvoyés par l'API FastAPI (backend/app/api/routes).

export type Source = {
  n: number;
  document: string; // « AI Act (règlement (UE) 2024/1689) »
  label: string; // « Article 99 »
  url: string;
  score: number | null; // score du reranker ; null pour une section jointe par citation
  cited_by: string | null;
  cited: boolean;
};

export type Verification = {
  cited: number[];
  invalid_markers: number[];
  quotes_checked: number;
  unverified_quotes: string[]; // extraits « ... » introuvables mot pour mot dans les sources
  uncited_sentences: string[];
  unsourced_facts: string[];
};

export type ChatEvent =
  | { type: "statut"; etape: "recherche" | "redaction" }
  | { type: "sources"; session_id: string; zone: string; sources: Source[] }
  | { type: "texte"; texte: string }
  | {
      type: "fin";
      session_id: string;
      answer: string;
      refused: boolean;
      zone: string;
      verification: Verification | null;
    };

export type ConversationSummary = {
  id: string;
  titre: string;
  created_at: string;
  updated_at: string;
};

export type StoredMessage = {
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  created_at: string;
};

export type QuestionOption = { valeur: string; libelle: string };

export type Question = {
  champ: string;
  question: string;
  aide: string | null;
  multiple: boolean;
  options: QuestionOption[];
};

export type Level =
  | "interdit"
  | "haut_risque"
  | "haut_risque_a_confirmer"
  | "incertain"
  | "obligations_specifiques"
  | "risque_minimal";

export type Report = {
  synthese?: string;
  classification: { niveau: Level };
  obligations: { id: string; titre?: string }[];
  avertissement: string;
  appels_llm?: number;
};

export type AnalysisEvent =
  | { type: "en_attente"; analyse_id: string; position: number }
  | { type: "debut"; analyse_id: string }
  | { type: "etape"; etape: string; libelle: string }
  | { type: "questions"; analyse_id: string; questions: Question[] }
  | { type: "rapport"; analyse_id: string; rapport: Report; markdown: string }
  | { type: "erreur"; message: string };

export type AnalysisStatus = "en_attente" | "en_cours" | "questions" | "termine" | "erreur";

export type AnalysisState = {
  analyse_id: string;
  statut: AnalysisStatus;
  titre: string;
  cree_le: string;
  questions: Question[];
  rapport: Report | null;
  markdown?: string | null;
};

export type AnalysisSummary = {
  id: string;
  titre: string;
  source: "description" | "document";
  statut: AnalysisStatus;
  niveau: Level | null;
  created_at: string;
  updated_at: string;
};

export type Ticket = {
  analyse_id: string;
  position: number;
  depuis: number;
  flux: string;
  document?: { nom: string; pages: number; pages_sans_texte: number[] };
};
