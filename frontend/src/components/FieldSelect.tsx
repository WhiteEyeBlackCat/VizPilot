import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Radix Select reserves the empty string for "no selection", so the "—"
// choice carries a sentinel that is mapped back to "" (and then to null in
// the ChartSpec) before it leaves this component.
const NONE = "__none__";

interface Props {
  label: string;
  value: string;
  onChange: (value: string) => void;
  choices: readonly string[];
  /** Offer a "—" (none) choice; the builder's optional axes need it. */
  allowNone?: boolean;
  disabled?: boolean;
  className?: string;
}

export function FieldSelect({
  label,
  value,
  onChange,
  choices,
  allowNone = true,
  disabled = false,
  className,
}: Props) {
  const inert = disabled || choices.length === 0;
  return (
    <div className={className ?? "flex flex-col gap-1"}>
      <Label className="text-xs font-normal text-muted-foreground">{label}</Label>
      <Select
        value={value === "" ? (allowNone ? NONE : "") : value}
        onValueChange={(v) => onChange(v === NONE ? "" : v)}
        disabled={inert}
      >
        <SelectTrigger aria-label={label} className="h-8 min-w-28 text-sm">
          <SelectValue placeholder="—" />
        </SelectTrigger>
        <SelectContent>
          {allowNone && <SelectItem value={NONE}>—</SelectItem>}
          {choices.map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
