import React, { useState, useEffect, useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Loader2 } from "lucide-react";
import UploadComponent from "./Upload";
import Prediction from "./Prediction";
import Cobb from "./cobb";
import "../App.css";

function Posture() {
  const [prediction, setPrediction] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showCobb, setShowCobb] = useState(false);

  const cobbRef = useRef(null);
  const { scrollYProgress } = useScroll();

  // Apply smooth scrolling animations
  const fadeIn = {
    hidden: { opacity: 0, y: 50 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleReUpload = () => {
    setPrediction("");
    setImage(null);
    setPreview(null);
    setShowCobb(false);
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("/api/predict-posture", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction({
        label: data.posture,
        confidence: data.confidence.toFixed(2),
      });
    } catch (error) {
      setPrediction({ label: "Error", confidence: "N/A" });
    } finally {
      setLoading(false);
    }
  };

  const handleCalculateCobb = () => {
    setShowCobb(true);
    setTimeout(() => {
      cobbRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  };

  return (
    <div className="scroll-container">
      {loading ? (
        <motion.div
          className="flex flex-col items-center justify-center mt-40 prediction"
          variants={fadeIn}
          initial="hidden"
          animate="visible"
        >
          <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
          <p className="mt-2 text-lg font-semibold">Processing...</p>
        </motion.div>
      ) : (
        <>
          {!prediction && (
            <motion.div variants={fadeIn} initial="hidden" animate="visible">
              <UploadComponent
                preview={preview}
                handleFileChange={handleFileChange}
                handleUpload={handleUpload}
                loading={loading}
                title="Posture"
              />
            </motion.div>
          )}

          {prediction && (
            <motion.div
              className="flex flex-col items-center"
              variants={fadeIn}
              initial="hidden"
              animate="visible"
            >
              <Prediction
                prediction={prediction}
                preview={preview}
                handleReUpload={handleReUpload}
                handleCalculateCobb={handleCalculateCobb}
              />
            </motion.div>
          )}

          {showCobb && (
            <motion.div
              ref={cobbRef}
              className="mt-20 mb-20"
              variants={fadeIn}
              initial="hidden"
              animate="visible"
            >
              <Cobb image={preview} handleBack={() => setShowCobb(false)} />
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}

export default Posture;
