import { TextInput, Badge, Stack } from "@mantine/core";
import type { Field } from "../types";

export function FieldEditor({ label, field, onChange }: { label: string; field: Field; onChange: (value: string) => void }) {
  return <Stack gap={3}>
    <TextInput label={label} value={field.value ?? ""} placeholder={field.status === "not_applicable" ? "Not applicable" : "Enter value"}
      onChange={(event) => onChange(event.currentTarget.value)} />
    {field.uncertain && <Badge color="yellow" variant="light">Uncertain{field.reason ? `: ${field.reason}` : ""}</Badge>}
  </Stack>;
}

