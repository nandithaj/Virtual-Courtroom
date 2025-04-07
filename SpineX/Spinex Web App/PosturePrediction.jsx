import React from "react";

const PosturePrediction = ({ posturePrediction, preview, handleReUpload }) => {
  // Log the prediction data
  console.log("Posture Prediction Data:", posturePrediction);
  console.log("Posture Label:", posturePrediction.posture);

  return (
    <div className="flex flex-row items-center predictionbox mt-24 space-x-10">
      {/* Uploaded Image */}
      {preview && (
        <img
          src={preview}
          alt="Uploaded preview"
          className="w-50 h-50 rounded-lg object-cover img1 mr-20"
        />
      )}

      {/* Prediction Output */}
      <div className="prediction mt-3">
        <h2 className="text-3xl font-bold text-[#28487B]">
          {posturePrediction.posture}
        </h2>
        <p className="text-lg mt-2">
          {posturePrediction.posture === "Normal"
            ? "The posture appears to be normal, with no abnormalities detected."
            : `The posture indicates the presence of ${posturePrediction.posture}.`}
        </p>
        <p className="text-lg mt-2 font-semibold">
          Confidence: {posturePrediction.confidence}%
        </p>

        {/* Re-upload Button */}
        <button
          onClick={handleReUpload}
          className="button mt-8 px-4 py-2 bg-[#28487B] text-[#B9CDEE] rounded-lg hover:bg-[#4468A6]"
        >
          Re-upload
        </button>
      </div>
    </div>
  );
};

export default PosturePrediction;
