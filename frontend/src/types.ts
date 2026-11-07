export type FieldStatus = "extracted" | "unreadable" | "missing" | "not_applicable";
export type Field = { value: string | null; status: FieldStatus; uncertain: boolean; reason?: string | null };
export type ExtractionResult = {
  label_type: "large" | "small" | "unknown";
  sample_id: Field; date_sampled: Field; date_received: Field; machine: Field;
  component: Field; serial_number: Field; reference_lube: Field; warnings: string[];
};
export type AnalyzeResponse = { success: boolean; model: string; result?: ExtractionResult; raw_output?: string; error?: string };

