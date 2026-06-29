"use client";

import { useCallback, useRef } from "react";
import dynamic from "next/dynamic";

const FluidSim = dynamic(() => import("./FluidSim"), { ssr: false });

const LINKS = [
  { href: "https://notes.zeyuanfeng.com",  label: "Notes",  desc: "Personal notes & thoughts" },
  { href: "https://resume.zeyuanfeng.com", label: "Resume", desc: "Work experience & skills"   },
  { href: "https://promo.zeyuanfeng.com",  label: "Promos", desc: "Referrals & promotions"     },
];

// Layout: hero sits at ~35% from top, cards at ~75%.
// If the waterline (0–1, where 0=top) is above a block → block is in the water (needs light text).
// If the waterline is below a block → block is in the air (needs dark text).
const HERO_Y  = 0.40;
const CARDS_Y = 0.78;

export default function HomePage() {
  const heroRef  = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);

  const onWaterline = useCallback((sy: number) => {
    const hero  = heroRef.current;
    const cards = cardsRef.current;
    if (!hero || !cards) return;

    // Hero text
    if (sy < HERO_Y) {
      // waterline is above hero → hero is submerged
      hero.style.setProperty("--title-color", "#e8f4ff");
      hero.style.setProperty("--sub-color",   "rgba(210,235,255,0.75)");
    } else {
      hero.style.setProperty("--title-color", "#1e293b");
      hero.style.setProperty("--sub-color",   "rgba(71,85,105,0.85)");
    }

    // Nav cards
    if (sy < CARDS_Y) {
      cards.style.setProperty("--card-bg",     "rgba(255,255,255,0.08)");
      cards.style.setProperty("--card-border", "rgba(255,255,255,0.18)");
      cards.style.setProperty("--card-label",  "#e8f4ff");
      cards.style.setProperty("--card-desc",   "rgba(190,220,255,0.65)");
    } else {
      cards.style.setProperty("--card-bg",     "rgba(255,255,255,0.60)");
      cards.style.setProperty("--card-border", "rgba(200,215,230,0.70)");
      cards.style.setProperty("--card-label",  "#1e293b");
      cards.style.setProperty("--card-desc",   "rgba(71,85,105,0.80)");
    }
  }, []);

  return (
    <main className="relative min-h-screen bg-white overflow-hidden">
      {/* Full-screen water sim */}
      <div className="absolute inset-0">
        <FluidSim onWaterline={onWaterline} />
      </div>

      {/* Overlay — pointer-events-none everywhere except cards */}
      <div className="relative z-10 flex flex-col min-h-screen pointer-events-none">
        {/* Hero */}
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
                style={{
                  background:  "var(--card-bg)",
                  border:      "1px solid var(--card-border)",
                }}
              >
                <p className="font-semibold" style={{ color: "var(--card-label)" }}>{label} →</p>
                <p className="text-sm mt-0.5" style={{ color: "var(--card-desc)" }}>{desc}</p>
              </a>
            ))}
          </div>
        </div>

        <p className="text-center text-xs pb-4 text-slate-300">
          © {new Date().getFullYear()} Zeyuan Feng
        </p>
      </div>
    </main>
  );
}
