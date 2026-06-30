"use client";

import { useCallback, useRef } from "react";
import dynamic from "next/dynamic";

const FluidSim = dynamic(() => import("./FluidSim"), { ssr: false });

const LINKS = [
  { href: "https://notes.zeyuanfeng.com",  label: "Notes",  desc: "Personal notes & thoughts" },
  { href: "https://resume.zeyuanfeng.com", label: "Resume", desc: "Work experience & skills"   },
  { href: "https://promo.zeyuanfeng.com",  label: "Promos", desc: "Referrals & promotions"     },
];

const HERO_Y  = 0.40;
const CARDS_Y = 0.78;

// Submerged palette — same in both modes (water is already dark blue).
const WATER = {
  title: "#e8f4ff",  sub: "rgba(210,235,255,0.75)",
  cardBg: "rgba(255,255,255,0.08)", cardBorder: "rgba(255,255,255,0.18)",
  cardLabel: "#e8f4ff", cardDesc: "rgba(190,220,255,0.65)",
};
// Air palette — queried fresh from matchMedia each callback (no state).
function airPalette(dark: boolean) {
  return dark
    ? { title: "#e2e8f0", sub: "rgba(148,163,184,0.85)",   cardBg: "rgba(0,0,0,0.40)",       cardBorder: "rgba(100,120,140,0.45)", cardLabel: "#e2e8f0", cardDesc: "rgba(148,163,184,0.75)" }
    : { title: "#1e293b", sub: "rgba(71,85,105,0.85)",     cardBg: "rgba(255,255,255,0.60)", cardBorder: "rgba(200,215,230,0.70)", cardLabel: "#1e293b", cardDesc: "rgba(71,85,105,0.80)"   };
}

export default function HomePage() {
  const heroRef  = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);

  // Read dark mode directly from matchMedia every frame — no React state,
  // no re-renders, always fresh.
  const onWaterline = useCallback((sy: number) => {
    const hero  = heroRef.current;
    const cards = cardsRef.current;
    if (!hero || !cards) return;

    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const air  = airPalette(dark);

    hero.style.setProperty("--title-color", sy < HERO_Y  ? WATER.title     : air.title);
    hero.style.setProperty("--sub-color",   sy < HERO_Y  ? WATER.sub       : air.sub);

    cards.style.setProperty("--card-bg",     sy < CARDS_Y ? WATER.cardBg     : air.cardBg);
    cards.style.setProperty("--card-border", sy < CARDS_Y ? WATER.cardBorder : air.cardBorder);
    cards.style.setProperty("--card-label",  sy < CARDS_Y ? WATER.cardLabel  : air.cardLabel);
    cards.style.setProperty("--card-desc",   sy < CARDS_Y ? WATER.cardDesc   : air.cardDesc);
  }, []);

  return (
    // bg-white dark:bg-black handled by Tailwind — no JS needed.
    <main className="relative min-h-screen overflow-hidden bg-white dark:bg-black">
      <div className="absolute inset-0">
        <FluidSim onWaterline={onWaterline} />
      </div>

      <div className="relative z-10 flex flex-col min-h-screen pointer-events-none">
        {/* Hero — initial colors set via Tailwind; JS overrides per-frame via CSS vars */}
        <div
          ref={heroRef}
          className="flex flex-col items-center justify-center flex-1 px-6 pb-12 text-center"
          style={{
            ["--title-color" as string]: "#1e293b",
            ["--sub-color"   as string]: "rgba(71,85,105,0.85)",
          }}
        >
          <h1
            className="text-5xl font-bold tracking-tight mb-2 transition-colors duration-300"
            style={{ color: "var(--title-color)" }}
          >
            Zeyuan Feng
          </h1>
          <p
            className="text-lg transition-colors duration-300"
            style={{ color: "var(--sub-color)" }}
          >
            Software Engineer
          </p>
        </div>

        {/* Cards */}
        <div className="px-6 pb-16">
          <div
            ref={cardsRef}
            className="max-w-lg mx-auto grid grid-cols-1 sm:grid-cols-3 gap-3 pointer-events-auto"
            style={{
              ["--card-bg"     as string]: "rgba(255,255,255,0.60)",
              ["--card-border" as string]: "rgba(200,215,230,0.70)",
              ["--card-label"  as string]: "#1e293b",
              ["--card-desc"   as string]: "rgba(71,85,105,0.80)",
            }}
          >
            {LINKS.map(({ href, label, desc }) => (
              <a
                key={href}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-2xl px-5 py-4 backdrop-blur-sm transition-all shadow-sm"
                style={{ background: "var(--card-bg)", border: "1px solid var(--card-border)" }}
              >
                <p className="font-semibold transition-colors duration-300" style={{ color: "var(--card-label)" }}>{label} →</p>
                <p className="text-sm mt-0.5 transition-colors duration-300" style={{ color: "var(--card-desc)" }}>{desc}</p>
              </a>
            ))}
          </div>
        </div>

        <p className="text-center text-xs pb-4 text-slate-300 dark:text-slate-700">
          © {new Date().getFullYear()} Zeyuan Feng
        </p>
      </div>
    </main>
  );
}
