import { ImageResponse } from "next/og";
import { siteConfig } from "@/config/site";

export const alt = `${siteConfig.name} — a personal universe`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/** Default social preview image: navy field, thin grid, yellow name. */
export default function OpengraphImage() {
  const lines = Array.from({ length: 16 }, (_, i) => i);
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", background: "#080F20", position: "relative", fontFamily: "sans-serif" }}>
        {lines.map((i) => (
          <div key={`v${i}`} style={{ position: "absolute", left: i * 80, top: 0, width: 1, height: 630, background: "#263650", opacity: 0.6 }} />
        ))}
        {lines.slice(0, 8).map((i) => (
          <div key={`h${i}`} style={{ position: "absolute", top: i * 80, left: 0, height: 1, width: 1200, background: "#263650", opacity: 0.6 }} />
        ))}
        <div style={{ position: "absolute", right: 120, top: 135, width: 360, height: 360, borderRadius: 999, border: "2px solid #4169E1", display: "flex" }} />
        <div style={{ position: "absolute", right: 190, top: 205, width: 220, height: 220, borderRadius: 999, border: "2px solid #E8FF47", display: "flex" }} />
        <div style={{ position: "absolute", right: 292, top: 307, width: 16, height: 16, borderRadius: 999, background: "#E8FF47", display: "flex" }} />
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 90px", height: "100%" }}>
          <div style={{ color: "#A5B2C8", fontSize: 26, letterSpacing: 6 }}>HELLO, I’M</div>
          <div style={{ color: "#E8FF47", fontSize: 120, fontWeight: 800, lineHeight: 1.05 }}>{siteConfig.firstName}</div>
          <div style={{ color: "#F5F5F0", fontSize: 32, marginTop: 24, maxWidth: 640 }}>Builder · AI · products · technology for social good</div>
        </div>
      </div>
    ),
    size,
  );
}
