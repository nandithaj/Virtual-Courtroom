import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Navbar from "./components/NavBar";
import Home from "./components/Home";
import Box from "./components/Box";
import Scoliosis from "./components/Scoliosis";
import Posture from "./components/Posture";
import "./App.css";

function App() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [currentPage, setCurrentPage] = useState("home");
  const [homeAnimationComplete, setHomeAnimationComplete] = useState(false); // Track Home animation

  useEffect(() => {
    const handleScroll = (event) => {
      if (currentPage !== "home") return;
      event.preventDefault();

      setScrollProgress((prev) => {
        let newValue = prev + (event.deltaY > 0 ? 0.05 : -0.05);
        return Math.min(1, Math.max(0, newValue));
      });
    };

    window.addEventListener("wheel", handleScroll, { passive: false });

    return () => {
      window.removeEventListener("wheel", handleScroll);
    };
  }, [currentPage]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentPage]);

  return (
    <>
      <Navbar
        onNavigate={() => {
          setCurrentPage("home");
          setHomeAnimationComplete(false); // Reset animation tracking
        }}
      />

      <div className="min-h-screen flex flex-col items-center justify-center overflow-hidden main">
        {currentPage === "home" ? (
          <div className="flex flex-col items-center justify-center min-h-screen mt-28 overflow-hidden">
            {/* Home Component Animation */}
            <motion.div
              initial={{ opacity: 1, y: 0 }}
              animate={{
                opacity: scrollProgress > 0.3 ? 0.6 : 1,
                y: scrollProgress > 0.3 ? -80 : 0,
              }}
              transition={{ duration: 0.5 }}
              onAnimationComplete={() => setHomeAnimationComplete(true)} // Trigger Box animation
              className="flex flex-col items-center justify-center min-h-[50vh]"
            >
              <Home />
            </motion.div>

            {/* Box Components Animation (only starts after Home animation is complete) */}
            {homeAnimationComplete && (
              <div className="flex space-x-10 mt-[-30px]">
                <motion.div
                  initial={{ y: 150, opacity: 0 }}
                  animate={{
                    y: (0.3 - scrollProgress) * 150,
                    opacity: scrollProgress,
                  }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  onClick={() => setCurrentPage("scoliosis")}
                >
                  <Box
                    title="Scoliosis Detection"
                    description="Check if you have Scoliosis"
                  />
                </motion.div>

                <motion.div
                  initial={{ y: 150, opacity: 0 }}
                  animate={{
                    y: (0.3 - scrollProgress) * 150,
                    opacity: scrollProgress,
                  }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  onClick={() => setCurrentPage("posture")}
                >
                  <Box
                    title="Posture Detection"
                    description="Check if you have Kyphosis or Lordosis"
                  />
                </motion.div>
              </div>
            )}
          </div>
        ) : (
          <>
            {currentPage === "scoliosis" && <Scoliosis />}
            {currentPage === "posture" && <Posture />}
          </>
        )}
      </div>
    </>
  );
}

export default App;
