import Image from "next/image";
import { GraduationCap } from "lucide-react";
import type { Education } from "@/lib/schemas";
import { Card } from "@/components/ui/card";
import { Placeholder } from "@/components/ui/placeholder";
import { formatPeriod } from "@/lib/dates";

function List({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h4 className="font-display text-sm font-semibold">{title}</h4>
      <ul className="mt-1.5 list-disc space-y-1 pl-4 text-sm text-muted marker:text-blue-soft">
        {items.map((i) => (
          <li key={i}>{i}</li>
        ))}
      </ul>
    </div>
  );
}

export function EducationCard({ education: e }: { education: Education }) {
  const period = [formatPeriod(e.start), formatPeriod(e.end)].filter(Boolean).join(" – ");
  const empty = !e.activities.length && !e.interests.length && !e.accomplishments.length;
  return (
    <Card as="article" interactive className="flex flex-col">
      <div className="flex items-start gap-4">
        <div className="grid h-12 w-12 shrink-0 place-items-center overflow-hidden rounded-lg border border-border bg-bg text-blue-soft">
          {e.logo ? <Image src={e.logo} alt="" width={48} height={48} className="object-contain p-1.5" /> : <GraduationCap size={20} aria-hidden="true" />}
        </div>
        <div className="min-w-0">
          <h3 className="font-display text-xl font-semibold leading-tight">{e.institution}</h3>
          {e.school && <p className="mt-0.5 text-sm text-blue-soft">{e.school}</p>}
          <p className="mt-2 text-[0.95rem]">{e.degree}</p>
          <p className="mt-1 font-mono text-xs text-muted">{period || <Placeholder inline hint="dates" />}</p>
        </div>
      </div>
      <div className="mt-5 space-y-4">
        <List title="Relevant activities" items={e.activities} />
        <List title="Academic interests" items={e.interests} />
        <List title="Selected accomplishments" items={e.accomplishments} />
        {empty && <Placeholder hint="activities, interests and accomplishments — add in src/data/education.ts" />}
      </div>
    </Card>
  );
}
