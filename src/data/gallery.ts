/**
 * Beyond the Desk galleries.
 *
 * Kitchen photos: put images in public/images/cooking/ and add an entry.
 * Dance: photos in public/images/dancing/, videos as YouTube/Vimeo ids or files.
 * Entries with `draft: true` only appear in local previews. The draft samples
 * below exist only so the layout can be previewed — replace them.
 */
import { danceItemSchema, kitchenItemSchema, parseList } from "@/lib/schemas";

const kitchenEntries = [
  {
    id: "sample-dish-1",
    dish: "Sample dish (placeholder)",
    image: { src: "/images/cooking/placeholder-1.svg", alt: "Placeholder pattern standing in for a food photograph", width: 800, height: 1000 },
    description: "Replace with a real photograph and dish.",
    draft: true,
  },
  {
    id: "sample-dish-2",
    dish: "Sample dish (placeholder)",
    image: { src: "/images/cooking/placeholder-2.svg", alt: "Placeholder pattern standing in for a food photograph", width: 800, height: 600 },
    draft: true,
  },
  /*
   * {
   *   id: "masala-chai",
   *   dish: "Dish name",
   *   image: { src: "/images/cooking/dish.jpg", alt: "Describe the photo", width: 1200, height: 1500 },
   *   description: "Optional one-liner",
   *   recipe: "Optional recipe text or a link",
   *   story: "Optional memory behind it",
   *   date: "2026-12-01",
   * },
   */
];

const danceEntries = [
  {
    id: "sample-performance",
    title: "Sample performance (placeholder)",
    style: "Style",
    occasion: "Occasion",
    image: { src: "/images/dancing/placeholder-1.svg", alt: "Placeholder pattern standing in for a dance photograph", width: 1200, height: 800 },
    draft: true,
  },
  /*
   * {
   *   id: "performance-id",
   *   title: "Performance name",
   *   style: "Dance style",
   *   occasion: "Annual showcase",
   *   date: "2025-04-12",
   *   video: { provider: "youtube", id: "VIDEO_ID", thumbnail: "/images/dancing/thumb.jpg" },
   *   choreography: "Optional choreography notes",
   *   description: "Optional description",
   * },
   */
];

export const kitchenItems = parseList(kitchenItemSchema, kitchenEntries, "src/data/gallery.ts (kitchen)");
export const danceItems = parseList(danceItemSchema, danceEntries, "src/data/gallery.ts (dance)");
