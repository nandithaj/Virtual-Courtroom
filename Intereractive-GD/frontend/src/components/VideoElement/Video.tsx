import React, { useRef, useState, useEffect } from 'react';
import { Video as VideoIcon, VideoOff } from 'lucide-react';
import axios from 'axios';
import { captureCompressedImage, uploadScreenshot } from '../../utils/imageUtils';

interface VideoProps {
  width?: string;
  height?: string;
  maxWidth?: string;
  topic?: string;
}

const Video: React.FC<VideoProps> = ({ 
  width = "100%", 
  height = "100%",
  maxWidth = "100%",
  topic = ""
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    startStream();
    
    // Clean up function
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const takeScreenshot = async () => {
      if (videoRef.current && isVideoOn) {
        try {
          console.log("Taking screenshot (every 30 seconds)...");
          const user = JSON.parse(localStorage.getItem("user") || "{}");
          if (!user || !user.user_id) {
            console.warn("Cannot take screenshot: No user ID found");
            return;
          }

          // Capture and compress the image
          const imageData = await captureCompressedImage(videoRef.current, 0.6, 640);
          console.log(`Screenshot captured, size: ${Math.round(imageData.length/1024)} KB`);
          
          // Upload the image
          console.log(`Uploading screenshot for user ${user.user_id}...`);
          await uploadScreenshot(imageData, user.user_id, topic || "");
          console.log("Screenshot upload complete");
        } catch (error: any) {
          const errorMessage = error?.message || String(error);
          console.error("Error processing screenshot:", errorMessage);
          // Display the error in the UI
          setError(`Screenshot error: ${errorMessage}`);
          // Clear error after 8 seconds
          setTimeout(() => setError(""), 8000);
        }
      }
    };
    
    // Only start taking screenshots if video is enabled
    let screenshotInterval: number | null = null;
    if (isVideoOn) {
      console.log("Setting up screenshot interval (30 seconds)");
      screenshotInterval = window.setInterval(takeScreenshot, 30000); // Every 30 seconds
      
      // Don't take an initial screenshot immediately - wait for the first interval
      // This prevents duplicates at startup
    }
    
    return () => {
      if (screenshotInterval) {
        console.log("Clearing screenshot interval");
        clearInterval(screenshotInterval);
      }
    };
  }, [topic, isVideoOn]);

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false // Audio handled by SpeechToText component
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("Could not access camera");
    }
  };

  const toggleVideo = async () => {
    if (streamRef.current) {
      const videoTrack = streamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoOn(!isVideoOn);
      }
    }
  };

  return (
    <div className="w-full" style={{ maxWidth }}>
      <div className="relative rounded-2xl overflow-hidden bg-gray-900 shadow-xl aspect-video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        {/* Video Overlay when video is off */}
        {!isVideoOn && (
          <div className="absolute inset-0 bg-gray-800 flex items-center justify-center">
            <VideoOff size={48} className="text-gray-400" />
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="absolute inset-0 bg-gray-800/90 flex items-center justify-center">
            <p className="text-white text-center px-4">{error}</p>
          </div>
        )}

        {/* Video Toggle */}
        {/* <button
          onClick={toggleVideo}
          className={`absolute bottom-4 right-4 p-3 rounded-full transition-all shadow-lg backdrop-blur-sm ${
            isVideoOn
              ? "bg-violet-600/90 hover:bg-violet-700"
              : "bg-red-500/90 hover:bg-red-600"
          }`}
        >
          {isVideoOn ? (
            <VideoIcon className="w-6 h-6 text-white" />
          ) : (
            <VideoOff className="w-6 h-6 text-white" />
          )}
        </button> */}
      </div>
    </div>
  );
};

export default Video;
