import React from "react";
import { useNavigate } from "react-router-dom";

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      {/* Header */}
      <header className="relative z-10 border-b border-white/10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-2xl font-bold text-white">
                Interactive GD
              </div>
            </div>
            {/* <nav className="hidden md:flex items-center gap-6">
              <a
                href="#pricing"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Pricing
              </a>
              <a
                href="#blog"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Blog
              </a>
              <a
                href="#affiliate"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Affiliate Program
                <span className="ml-1 inline-flex items-center rounded-md bg-yellow-400/10 px-2 py-1 text-xs font-medium text-yellow-500 ring-1 ring-inset ring-yellow-400/20">
                  New
                </span>
              </a>
            </nav> */}
            <div className="flex items-center gap-4">
              <button
                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                onClick={() => navigate("/signin")}
              >
                Sign in
              </button>
              <button
                className="px-4 py-2 text-sm bg-yellow-500 text-black rounded-md hover:bg-yellow-400 transition-colors"
                onClick={() => navigate("/get-started")}
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 py-24">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-start gap-8 max-w-[720px]">
            <div className="inline-flex items-center rounded-full border border-yellow-500/20 bg-yellow-500/10 px-3 py-1 text-sm text-yellow-500">
              <span className="mr-2">✨</span> One-stop solution to practice
              Group Discussions for your interviews
            </div>
            <h1 className="text-5xl font-bold tracking-tight lg:text-7xl">
              Master <span className="text-yellow-500">Group</span>
              <br />
              <span className="text-yellow-500">Discussions with</span>
              <br />
              Confidence
            </h1>
            <p className="text-xl text-gray-400">
              Practice, perform, and perfect your group discussion skills with
              our AI-driven interactive platform—where virtual participants meet
              real-world impact!
            </p>
            <button
              className="px-6 py-3 text-lg bg-yellow-500 text-black rounded-md hover:bg-yellow-400 transition-colors"
              onClick={() => navigate("/get-started")}
            >
              Get Started →
            </button>
          </div>
        </div>
      </section>

      {/* Video Examples Grid */}
      {/* <section className="relative z-10 py-24">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="relative overflow-hidden rounded-2xl bg-gray-900/50 aspect-[4/3]"
              >
                <img
                  src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Screenshot%202025-01-19%20at%201.19.25%E2%80%AFPM-WjCYUOSq3SYj4RwzmbzagAPoweAXhp.png"
                  alt={`Video example ${i}`}
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4 right-4">
                  <h3 className="text-lg font-semibold text-white">
                    Video Title
                  </h3>
                  <p className="text-sm text-gray-400">Description goes here</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section> */}

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 py-8">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-400">
              © 2025 Interactive GD. All rights reserved.
            </p>
            {/* <div className="flex gap-4">
              <a
                href="#privacy"
                className="text-sm text-gray-400 hover:text-white"
              >
                Privacy
              </a>
              <a
                href="#terms"
                className="text-sm text-gray-400 hover:text-white"
              >
                Terms
              </a>
            </div> */}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
