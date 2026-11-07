import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Container, FileInput, Group, JsonInput, SimpleGrid, Stack, Text, TextInput, Title } from "@mantine/core";
import { analyze } from "./services/api";
import { FieldEditor } from "./components/FieldEditor";
import type { ExtractionResult } from "./types";

const fields = [["sample_id", "Sample ID"], ["date_sampled", "Date Sampled"], ["date_received", "Date Received"],
  ["machine", "Machine"], ["component", "Component"], ["serial_number", "Serial Number"], ["reference_lube", "Reference Lube"]] as const;
type FieldKey = typeof fields[number][0];

function csvValue(value: string | null): string {
  const safe = value ?? "";
  const protectedValue = /^[=+\-@]/.test(safe) ? `'${safe}` : safe;
  return `"${protectedValue.replaceAll('"', '""')}"`;
}

export function downloadCsv(result: ExtractionResult) {
  const headers = ["Sample ID", "Date Sampled", "Date Received", "Machine", "Component", "Serial Number", "Refrence Lube"];
  const values = fields.map(([key]) => result[key].value);
  const blob = new Blob([[headers, values].map((row) => row.map(csvValue).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
  const name = window.prompt("CSV filename", "mrg-sample.csv") || "mrg-sample.csv";
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = name.endsWith(".csv") ? name : `${name}.csv`; link.click();
  URL.revokeObjectURL(link.href);
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [status, setStatus] = useState("File selected");
  const [error, setError] = useState("");
  const [raw, setRaw] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const json = useMemo(() => result ? JSON.stringify(result, null, 2) : "", [result]);
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const update = (key: FieldKey, value: string) => result && setResult({ ...result, [key]: { ...result[key], value: value || null, status: value ? "extracted" : "missing" } });

  async function submit() {
    if (!file || isSubmitting) return setError("Choose an MP4 or MOV video first.");
    setError(""); setIsSubmitting(true); setStatus("Analyzing");
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 240_000);
    try {
      const response = await analyze(file, date, controller.signal);
      if (!response.result) throw new Error("Gemini returned no validated extraction.");
      setResult(response.result); setRaw(response.raw_output || ""); setStatus("Completed");
    } catch (exception) {
      setStatus("Failed");
      setError(exception instanceof DOMException && exception.name === "AbortError"
        ? "Analysis timed out after 240 seconds. No second request was started."
        : exception instanceof Error ? exception.message : "Analysis failed.");
    } finally {
      window.clearTimeout(timeout); setIsSubmitting(false);
    }
  }

  return <Container size="lg" py="xl"><Title order={1}>MRG Sample Scanner — Gemini Prototype</Title><Text c="dimmed" mb="lg">Real Gemini extraction only; no fabricated results.</Text>
    <SimpleGrid cols={{ base: 1, md: 2 }}>
      <Card withBorder><Stack>
        <FileInput label="Sample video" placeholder="MP4 or MOV" accept="video/mp4,video/quicktime" value={file} onChange={(value) => { setFile(value); setStatus(value ? "File selected" : ""); }} />
        {previewUrl && <video controls src={previewUrl} style={{ width: "100%" }} />}
        <TextInput label="Receiving date" type="date" value={date} onChange={(e) => setDate(e.currentTarget.value)} />
        <Button onClick={submit} loading={isSubmitting} disabled={!file || isSubmitting}>Analyze with Gemini</Button>
        <Text>Status: {status === "Analyzing" ? "Analyzing video, then validating structured fields…" : status}</Text>
        {error && <Alert color="red" title="Analysis error">{error}</Alert>}
      </Stack></Card>
      <Card withBorder><Stack>
        <Title order={3}>Reviewed extraction</Title>
        {result ? <><Text>Label type: <b>{result.label_type}</b></Text>
          <Alert color="yellow" title="Manual verification required">
            Verify handwritten identifiers, especially Sample ID, Date Sampled, and Serial Number,
            plus the physically circled component number and GS/NGS selection against the video.
          </Alert>
          {result.warnings.map((warning) => <Alert key={warning} color="yellow">{warning}</Alert>)}
          {fields.map(([key, label]) => <FieldEditor key={key} label={label} field={result[key]} onChange={(value) => update(key, value)} />)}
          <Group><Button onClick={() => downloadCsv(result)}>Download CSV</Button></Group>
          <JsonInput label="JSON preview" value={json} readOnly autosize minRows={5} /><Text size="xs" c="dimmed">Raw model output retained for diagnostics: {raw ? "yes" : "no"}</Text>
        </> : <Text c="dimmed">Validated results will appear here after analysis.</Text>}
      </Stack></Card>
    </SimpleGrid>
  </Container>;
}
