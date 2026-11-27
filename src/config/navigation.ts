export interface NavItem {
  href: string;
  label: string;
  /** Short description used in the mobile menu and metadata. */
  description: string;
}

/** The six primary pages. Order here is the order in the header. */
export const primaryNav: NavItem[] = [
  { href: "/", label: "Hello", description: "Where it all begins" },
  { href: "/work", label: "Work", description: "Experience, recognition and education" },
  { href: "/projects", label: "Projects", description: "A technical implementation library" },
  { href: "/wonder-and-purpose", label: "Wonder & Purpose", description: "Questions I keep coming back to" },
  { href: "/learn", label: "Learn", description: "A living record of what I'm learning" },
  { href: "/beyond-the-desk", label: "Beyond the Desk", description: "Food, dance and writing" },
];
