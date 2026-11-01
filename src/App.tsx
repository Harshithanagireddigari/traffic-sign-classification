import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Demo from "@/components/Demo";
import Dashboard from "@/components/Dashboard";
import CodeSection from "@/components/CodeSection";
import Concepts from "@/components/Concepts";
import RunLocally from "@/components/RunLocally";
import Footer from "@/components/Footer";
import { ModelProvider } from "@/lib/model";

export default function App() {
  return (
    <ModelProvider>
      <div className="min-h-screen bg-slate-950 font-sans text-slate-200">
        <Navbar />
        <main>
          <Hero />
          <Demo />
          <Dashboard />
          <CodeSection />
          <Concepts />
          <RunLocally />
        </main>
        <Footer />
      </div>
    </ModelProvider>
  );
}
