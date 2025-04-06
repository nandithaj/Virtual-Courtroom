import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import UploadComponent from "./Upload";
import Prediction from "./Prediction";
import Cobb from "./cobb";
import TreatmentRecommendations from "./TreatmentRecommendations";
import "../App.css";

function Scoliosis() {
  const [prediction, setPrediction] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showCobb, setShowCobb] = useState(false);
  const [showRecommendation, setShowRecommendation] = useState(false);
  const [angleResult, setAngleResult] = useState(null);

  const cobbRef = useRef(null);
  const treatmentRef = useRef(null);

  // Animation for smooth fade-in
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
    setShowRecommendation(false);
  };

  const [uploadedFilename, setUploadedFilename] = useState("");

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      setPrediction({
        label: data.prediction,
        confidence: (data.confidence * 100).toFixed(2),
      });

      // Save the filename to use later for segmentation
      setUploadedFilename(image);
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

  const handleGetRecommendation = (angleData) => {
    setAngleResult(angleData);
    setShowRecommendation(true);

    // Scroll into view down to the recommendation component
    setTimeout(() => {
      treatmentRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
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
                title="Scoliosis"
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
              <Cobb
                image={preview} // For displaying the image
                imageFile={image} // 👈 Send actual file for processing
                uploadedFilename={uploadedFilename}
                setUploadedFilename={setUploadedFilename}
                handleBack={() => setShowCobb(false)}
                handleGetRecommendation={handleGetRecommendation}
              />
              {/* <button
                onClick={() => handleGetRecommendation(calculatedAngle)}
                className="btn-primary"
              ></button> */}
            </motion.div>
          )}

          {showRecommendation &&
            angleResult !== null &&
            !isNaN(angleResult) && (
              <motion.div
                ref={treatmentRef}
                className="mt-10 mb-20 flex justify-center"
                variants={fadeIn}
                initial="hidden"
                animate="visible"
              >
                <TreatmentRecommendations angleResult={angleResult} />
              </motion.div>
            )}
        </>
      )}
    </div>
  );
}

export default Scoliosis;
