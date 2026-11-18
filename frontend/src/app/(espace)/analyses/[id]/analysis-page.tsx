"use client";

import { useParams } from "next/navigation";

import { AnalysisView } from "@/components/analysis-view";

export function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  return <AnalysisView key={id} id={id} />;
}
