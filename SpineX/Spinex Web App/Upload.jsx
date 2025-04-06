import React from "react";
import { Upload } from "lucide-react"; // Upload icon from lucide-react

const UploadComponent = ({
  preview,
  handleFileChange,
  handleUpload,
  loading,
  title,
}) => {
  return (
    <div className="scontainer flex flex-col items-center pt-20 mt-10 space-y-5">
      <div className="hello">
        <h2 className="heading head text-3xl mt-5">{title} Classification</h2>

        {/* Custom Upload Box */}
        <label
          className="w-100 h-60 flex flex-col items-center justify-center border-2 border-dashed border-gray-400 rounded-lg cursor-pointer hover:border-blue-500 transition mt-8 relative"
          style={{
            backgroundImage: preview ? `url(${preview})` : "none",
            backgroundSize: "cover",
            backgroundPosition: "center",
            overflow: "hidden",
          }}
        >
          {/* Overlay for transparency */}
          {preview && (
            <div className="absolute inset-0 bg-black opacity-40 rounded-lg"></div>
          )}

          {/* Upload Icon and Text */}
          <div className="relative z-10 flex flex-col items-center">
            <Upload className="w-10 h-10 text-white" />
            <span className="text-white mt-2">Upload Image</span>
          </div>

          {/* Hidden Input */}
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        {/* Predict Button */}
        <button
          onClick={handleUpload}
          className="button mt-10 px-4 py-2 bg-[#28487B] text-[#B9CDEE] rounded-lg hover:bg-[#4468A6]"
          disabled={loading} // Disable button while loading
        >
          Predict
        </button>
      </div>
    </div>
  );
};

export default UploadComponent;
