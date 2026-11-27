import { Container } from "@/components/layout/container";
import { Portal } from "@/components/illustrations/primitives";
import { LinkButton } from "@/components/ui/button";

export default function NotFound() {
  return (
    <Container className="grid min-h-[70vh] items-center gap-10 py-16 md:grid-cols-2">
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-accent">404 · uncharted space</p>
        <h1 className="mt-4 font-display text-4xl font-semibold tracking-tight sm:text-5xl">This corner of the universe doesn’t exist (yet).</h1>
        <p className="mt-4 text-muted">The page may have moved, or it hasn’t been written. Let’s get you back to familiar ground.</p>
        <LinkButton href="/" variant="primary" className="mt-8">
          Back to the start
        </LinkButton>
      </div>
      <Portal className="mx-auto w-full max-w-sm" />
    </Container>
  );
}
