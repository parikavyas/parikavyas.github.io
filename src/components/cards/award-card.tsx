import { Award as AwardIcon } from "lucide-react";
import type { Award } from "@/lib/schemas";
import { Card } from "@/components/ui/card";
import { Placeholder } from "@/components/ui/placeholder";

export function AwardCard({ award }: { award: Award }) {
  return (
    <Card as="li" interactive className="flex gap-4 p-5">
      <AwardIcon size={18} className="mt-0.5 shrink-0 text-accent" aria-hidden="true" />
      <div className="min-w-0">
        <h3 className="font-display text-[1.02rem] font-semibold leading-snug">{award.name}</h3>
        <p className="mt-1 font-mono text-xs text-muted">
          {[award.year, award.issuer].filter(Boolean).join(" · ") || (
            <Placeholder inline hint="year & issuer — verify, then add in src/data/awards.ts" />
          )}
        </p>
        {award.description && <p className="mt-2 text-sm text-muted">{award.description}</p>}
      </div>
    </Card>
  );
}
