import React from "react";

const Prediction = ({
  prediction,
  preview,
  handleReUpload,
  handleCalculateCobb,
}) => {
  // Log the prediction data
  console.log("Prediction Data:", prediction);
  console.log("Prediction Label:", prediction.label);

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
          {prediction.label}
        </h2>
        <p className="text-lg mt-2">
          {prediction.label === "Normal"
            ? "The image appears to be normal."
            : `The image indicates the presence of ${prediction.label}.`}
        </p>
        <p className="text-lg mt-2 font-semibold">
          Confidence: {prediction.confidence}%
        </p>

        {/* Re-upload Button */}
        <button
          onClick={handleReUpload}
          className="button mt-8 px-4 py-2 bg-[#28487B] text-[#B9CDEE] rounded-lg hover:bg-[#4468A6]"
        >
          Re-upload
        </button>

        {/* Show Calculate Cobb Angle button only for Scoliosis */}
        {prediction.label === "Scoliosis" && (
          <button
            onClick={handleCalculateCobb} // Call the function passed from parent
            className="ml-4 mt-8 px-4 py-2 bg-[#28487B] text-[#B9CDEE] rounded-lg hover:bg-[#4468A6]"
          >
            Calculate Cobb Angle
          </button>
        )}
      </div>
    </div>
  );
};

export default Prediction;
