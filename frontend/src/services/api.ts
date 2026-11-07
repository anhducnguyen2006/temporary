import type { AnalyzeResponse } from "../types";

export async function analyze(video: File, receivingDate: string): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video", video);
  form.append("receiving_date", receivingDate);
  const response = await fetch("/api/analyze", { method: "POST", body: form });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Analysis failed (${response.status}).`);
  return body as AnalyzeResponse;
}

