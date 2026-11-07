import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Projects · Zeyuan Feng",
  description: "Side projects and small web experiments by Zeyuan Feng",
};

const PROJECTS = [
  {
    href: "/project/movie-soundtrack-quiz",
    title: "影史经典原声盲听挑战",
    subtitle: "Movie Soundtrack Blind Quiz",
    desc: "从 100 首殿堂级电影原声中随机抽 10 首盲听作答，通关生成六维鉴赏雷达图。",
    tags: ["Game", "Music", "iTunes API"],
  },
];

export default function ProjectsPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-16">
        <a href="/" className="text-sm text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 transition">
          ← Home
        </a>
        <h1 className="mt-4 text-3xl font-bold tracking-tight">Projects</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">Side projects and small web experiments.</p>

        <ul className="mt-10 grid gap-4">
          {PROJECTS.map((p) => (
            <li key={p.href}>
              <a
                href={p.href}
                className="block rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm transition hover:border-slate-400 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-600"
              >
                <p className="font-semibold">{p.title} →</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">{p.subtitle}</p>
                <p className="mt-2 text-sm text-slate-700 dark:text-slate-300">{p.desc}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {p.tags.map((t) => (
                    <span key={t} className="text-xs rounded-full bg-slate-100 px-2 py-0.5 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      {t}
                    </span>
                  ))}
                </div>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}
