import dynamic from "next/dynamic";

// HomePage contains "use client" hooks so we load it dynamically with ssr:false.
const HomePage = dynamic(() => import("./components/HomePage"), { ssr: false });

export default function Page() {
  return <HomePage />;
}
