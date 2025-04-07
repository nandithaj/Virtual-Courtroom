import React, { useEffect, useState } from "react";

interface AIResponseHandlerProps {
  userSpeech: string;
}

const ChatgptResponse: React.FC<AIResponseHandlerProps> = ({ userSpeech }) => {
  const [isProcessing, setIsProcessing] = useState(false);

  const generateAndSpeakResponse = async (text: string) => {
    if (!text.trim()) return;

    try {
      setIsProcessing(true);

      // Get response from your Flask backend
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });
      console.log(response);

      if (!response.ok) {
        throw new Error("Failed to get AI response");
      }

      const data = await response.json();
      const aiResponse = data.response;

      // Speak the response
      const utterance = new SpeechSynthesisUtterance(aiResponse);

      // Get available voices
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        utterance.voice = voices[1]; // Use a different voice than default
      }

      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error("Error in AI response:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (userSpeech) {
      generateAndSpeakResponse(userSpeech);
    }
  }, [userSpeech]);

  return <div>{isProcessing && <p>Processing your input...</p>}</div>;
};

export default ChatgptResponse;
