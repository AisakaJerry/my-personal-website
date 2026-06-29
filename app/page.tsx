import dynamic from "next/dynamic";

const FluidSim = dynamic(() => import("./components/FluidSim"), { ssr: false });

const LINKS = [
  { href: "https://notes.zeyuanfeng.com",  label: "Notes",  desc: "Personal notes & thoughts" },
  { href: "https://resume.zeyuanfeng.com", label: "Resume", desc: "Work experience & skills"   },
  { href: "https://promo.zeyuanfeng.com",  label: "Promos", desc: "Referrals & promotions"     },
];

export default function Home() {
  return (
    <main className="relative min-h-screen bg-white overflow-hidden">
      {/* Water sim — full background */}
      <div className="absolute inset-0">
        <FluidSim />
      </div>

      {/* Content overlay — pointer-events-none so clicks reach the canvas */}
      <div className="relative z-10 flex flex-col min-h-screen pointer-events-none">
        {/* Hero */}
        <div className="flex flex-col items-center justify-center flex-1 px-6 pt-24 pb-12 text-center">
          <h1 className="text-5xl font-bold tracking-tight mb-2 text-slate-800 drop-shadow-sm">
            Zeyuan Feng
          </h1>
          <p className="text-slate-500 text-lg">
            Software Engineer
          </p>
        </div>

        {/* Subdomain cards */}
        <div className="px-6 pb-16">
          <div className="max-w-lg mx-auto grid grid-cols-1 sm:grid-cols-3 gap-3 pointer-events-auto">
            {LINKS.map(({ href, label, desc }) => (
              <a
                key={href}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="group rounded-2xl border border-slate-200/60 bg-white/60 backdrop-blur-sm px-5 py-4 hover:bg-white/80 hover:border-slate-300 transition-all shadow-sm"
              >
                <p className="font-semibold text-slate-700 group-hover:text-slate-900">{label} →</p>
                <p className="text-slate-400 text-sm mt-0.5">{desc}</p>
              </a>
            ))}
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-slate-300 text-xs pb-4">
          © {new Date().getFullYear()} Zeyuan Feng
        </p>
      </div>
    </main>
  );
}
