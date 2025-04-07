"use client";

import { useState, useEffect } from "react";
import {
  BookOpen,
  Calendar,
  ChevronRight,
  Clock,
  MessageCircle,
  Mic,
  Star,
  TrendingUp,
  Users,
} from "lucide-react";
import GDEvaluation from "./GDEvaluation";
import ScreenshotEvaluation from "./ScreenshotEvaluation";
import SpeakingTimeScore from "./SpeakingTimeScore";
import GrammarEvaluation from './GrammarEvaluation';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [userId, setUserId] = useState<string | null>(null);
  const [speakingStats, setSpeakingStats] = useState<{
    average_percentage: number;
    total_sessions: number;
  }>({
    average_percentage: 0,
    total_sessions: 0
  });

  useEffect(() => {
    // Get user data from localStorage when component mounts
    const userData = localStorage.getItem("user");
    if (userData) {
      try {
        const parsedUser = JSON.parse(userData);
        if (parsedUser.user_id) {
          setUserId(parsedUser.user_id);
          // Fetch speaking stats
          fetchSpeakingStats(parsedUser.user_id);
        }
      } catch (error) {
        console.error("Error parsing user data:", error);
      }
    }
  }, []); // Empty dependency array means this runs once on mount

  const fetchSpeakingStats = async (userId: string) => {
    try {
      const response = await fetch(`http://localhost:8080/api/user/speaking-stats/${userId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setSpeakingStats({
            average_percentage: data.average_percentage,
            total_sessions: data.total_sessions
          });
        }
      }
    } catch (error) {
      console.error("Error fetching speaking stats:", error);
    }
  };

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
                href="#dashboard"
                className="text-sm text-yellow-500 hover:text-yellow-400 transition-colors"
              >
                Dashboard
              </a>
              <a
                href="#practice"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Practice
              </a>
              <a
                href="#history"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                History
              </a>
            </nav> */}
            <div className="flex items-center gap-4">
              <div className="absolute top-4 right-4 flex items-center space-x-3">
                <img
                  src={
                    JSON.parse(localStorage.getItem("user") || "{}")?.photo_url
                  }
                  alt="User profile"
                  className="w-10 h-10 rounded-full border-2 border-yellow-500"
                />
                <span className="font-semibold text-white">
                  {JSON.parse(localStorage.getItem("user") || "{}")?.name}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col gap-8">
            {/* Welcome Section */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <h1 className="text-3xl font-bold">
                  Welcome back,{" "}
                  {JSON.parse(localStorage.getItem("user") || "{}")?.name}!
                </h1>
                <p className="text-gray-400 mt-1">
                  Your GD skills are improving. Keep practicing!
                </p>
              </div>
              <button className="px-4 py-2 bg-yellow-500 hover:bg-yellow-400 text-black rounded-md transition-colors">
                Start New GD Session
              </button>
            </div>

            {userId && <GDEvaluation userId={userId} />}
            {userId && <ScreenshotEvaluation userId={userId} />}
            {userId && <SpeakingTimeScore userId={userId} />}
            {userId && <GrammarEvaluation userId={userId} />}

            {/* Tabs */}
            
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 py-8 mt-12">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-400">
              © 2025 Interactive GD. All rights reserved.
            </p>
            <div className="flex gap-4">
              <a
                href="#help"
                className="text-sm text-gray-400 hover:text-white"
              >
                Help
              </a>
              <a
                href="#settings"
                className="text-sm text-gray-400 hover:text-white"
              >
                Settings
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
