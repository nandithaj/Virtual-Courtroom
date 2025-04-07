// Add these type declarations at the top of the file
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
  namespace NodeJS {
    interface Timeout {}
  }
}

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Mic, MicOff, Save, AlertCircle, Hand } from "lucide-react";

interface SpeechToTextProps {
  sessionId?: string;
  participantId?: string;
  topic?: string;
  canLLMsStart?: boolean;
}

const SpeechToText: React.FC<SpeechToTextProps> = ({
  sessionId,
  participantId = "user1", // Default value, should be provided by auth system
  topic = "",
  canLLMsStart = false,
}) => {
  const [isListening, setIsListening] = useState(false);
  const [isHandRaised, setIsHandRaised] = useState(false);
  const [speechText, setSpeechText] = useState<string>("");
  const [recognition, setRecognition] = useState<any>(null);
  const [error, setError] = useState<string>("");
  const audioRef = useRef<HTMLAudioElement>(null);
  const speakingStartTimeRef = useRef<number | null>(null);
  const totalSpeakingTimeRef = useRef<number>(0);
  const sessionStartTimeRef = useRef<number>(Date.now());
  
  // Add conversation history tracking
  const conversationHistoryRef = useRef<Array<{role: string, content: string}>>([]);
  
  // Add a ref to track the actual recognition state
  const recognitionStateRef = React.useRef<boolean>(false);

  // Add a ref to track which LLM endpoint to use
  const llmEndpointRef = useRef<'llm1' | 'llm2'>('llm1');

  const [isAISpeaking, setIsAISpeaking] = useState(false);
  const aiSpeakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [currentParticipant, setCurrentParticipant] = useState<number | null>(null);
  
  // Add a lock to prevent multiple LLMs from interacting at the same time
  const llmLockRef = useRef<boolean>(false);

  // Function to update conversation history
  const updateConversationHistory = (role: string, content: string) => {
    conversationHistoryRef.current = [...conversationHistoryRef.current, { role, content }];
    // Keep only last 10 messages to maintain context without overwhelming
    if (conversationHistoryRef.current.length > 10) {
      conversationHistoryRef.current = conversationHistoryRef.current.slice(-10);
    }
  };

  // Effect to start LLM conversation when timer ends
  useEffect(() => {
    if (canLLMsStart && !isHandRaised) {
      console.log("Starting initial conversation");
      
      // Start the conversation with LLM1
      const startConversation = async () => {
        try {
          // Send initial message to start the discussion
          await sendToLLM("Let's begin the discussion about " + topic);
        } catch (error) {
          console.error('Error starting conversation:', error);
          setError('Failed to start the conversation');
        }
      };
      startConversation();
    }
  }, [canLLMsStart, topic, isHandRaised, isListening]);

  useEffect(() => {
    // Initialize speech recognition
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      try {
        const recognitionInstance = new SpeechRecognition();

        recognitionInstance.continuous = true;
        recognitionInstance.interimResults = true;
        recognitionInstance.lang = "en-US";

        recognitionInstance.onstart = () => {
          recognitionStateRef.current = true;
          setIsListening(true);
          setError("");
        };

        recognitionInstance.onresult = async (event: any) => {
          let finalTranscript = "";

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript + " ";
            }
          }

          if (finalTranscript) {
            setSpeechText((prev) => prev + finalTranscript);
          }
        };

        recognitionInstance.onerror = (event: any) => {
          console.error("Speech recognition error:", event.error);
          // Don't show error for aborted cases (manual stop)
          if (event.error !== 'aborted') {
            setError(`Error: ${event.error}`);
          }
          recognitionStateRef.current = false;
          setIsListening(false);
        };

        recognitionInstance.onend = () => {
          recognitionStateRef.current = false;
          setIsListening(false);
          
          // Update speaking time when recognition ends
          if (speakingStartTimeRef.current) {
            const speakingDuration = (Date.now() - speakingStartTimeRef.current) / 1000;
            totalSpeakingTimeRef.current += speakingDuration;
            speakingStartTimeRef.current = null;
          }
        };

        setRecognition(recognitionInstance);
      } catch (err) {
        setError("Failed to initialize speech recognition");
        console.error(err);
      }
    } else {
      setError("Speech recognition is not supported in this browser");
    }

    // Cleanup function
    return () => {
      if (recognition) {
        recognition.stop();
        recognitionStateRef.current = false;
      }
    };
  }, []);

  const toggleListening = useCallback(() => {
    if (!recognition) {
      setError("Speech recognition is not initialized");
      return;
    }

    try {
      if (recognitionStateRef.current) {
        recognition.stop();
        if (speakingStartTimeRef.current) {
          const speakingDuration = (Date.now() - speakingStartTimeRef.current) / 1000;
          totalSpeakingTimeRef.current += speakingDuration;
          console.log('Updated speaking time in toggleListening:', totalSpeakingTimeRef.current);
          speakingStartTimeRef.current = null;
        }
      } else {
        recognition.start();
        speakingStartTimeRef.current = Date.now();
        console.log('Started speaking time tracking');
      }
    } catch (err) {
      console.error(err);
      setError("Failed to toggle speech recognition");
      setIsListening(false);
      recognitionStateRef.current = false;
    }
  }, [recognition]);

  const saveTranscriptToBackend = async (text: string) => {
    if (!sessionId) return;

    try {
      const response = await fetch(
        `http://localhost:5000/sessions/${sessionId}/transcripts`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            participantId,
            text,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to save transcript");
      }

      setSpeechText("");
    } catch (err) {
      setError("Failed to save transcript to server");
      console.error(err);
    }
  };

  const handleSave = async () => {
    try {
      const blob = new Blob([speechText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "speech-text.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      await saveTranscriptToBackend(speechText);
    } catch (err) {
      setError("Failed to save text");
      console.error(err);
    }
  };

  const handleHandRaise = async () => {
    if (!isHandRaised) {
      // Only allow raising hand if AI is not speaking
      if (isAISpeaking) {
        setError("Please wait for the AI to finish speaking");
        return;
      }

      // Raise hand and stop any ongoing AI speech
      setIsHandRaised(true);
      setIsAISpeaking(false);
      setSpeechText(""); // Clear any previous speech
      
      // Clear any pending AI timeouts
      if (aiSpeakingTimeoutRef.current) {
        clearTimeout(aiSpeakingTimeoutRef.current);
        aiSpeakingTimeoutRef.current = null;
      }
      
      // Stop any ongoing AI speech
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
      
      // Stop browser speech synthesis if active
      window.speechSynthesis.cancel();
      
      // Release the LLM lock if it was held
      llmLockRef.current = false;
      
      // Start listening immediately when hand is raised
      if (recognition && !recognitionStateRef.current) {
        recognition.start();
      }
    } else {
      // Lower hand and stop listening
      setIsHandRaised(false);
      if (recognition && recognitionStateRef.current) {
        recognition.stop();
        
        // Update speaking time when stopping recognition
        if (speakingStartTimeRef.current) {
          const speakingDuration = (Date.now() - speakingStartTimeRef.current) / 1000;
          totalSpeakingTimeRef.current += speakingDuration;
          speakingStartTimeRef.current = null;
        }
      }
      
      // Store the complete speech in MongoDB
      if (speechText.trim()) {
        const user = JSON.parse(localStorage.getItem("user") || "{}");
        if (user && user.user_id) {
          try {
            // Calculate total session duration and speaking time
            const totalDuration = (Date.now() - sessionStartTimeRef.current) / 1000;
            if (speakingStartTimeRef.current) {
              const speakingDuration = (Date.now() - speakingStartTimeRef.current) / 1000;
              totalSpeakingTimeRef.current += speakingDuration;
            }

            console.log('Speaking time before saving:', totalSpeakingTimeRef.current);
            console.log('Total duration:', totalDuration);

            // Store speech text
            const speechResponse = await fetch('http://localhost:8080/api/user/speech', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                user_id: user.user_id,
                text: speechText.trim(),
                topic: topic
              }),
            });

            if (!speechResponse.ok) {
              const errorData = await speechResponse.json();
              console.error('Error saving speech to database:', errorData);
              setError('Failed to save speech to database');
            }

            // Store speaking time
            const speakingTimeResponse = await fetch('http://localhost:8080/api/user/speaking-time', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                user_id: user.user_id,
                session_id: sessionId || Date.now().toString(),
                speaking_duration: totalSpeakingTimeRef.current,
                total_duration: totalDuration
              }),
            });

            if (!speakingTimeResponse.ok) {
              const errorData = await speakingTimeResponse.json();
              console.error('Error saving speaking time to database:', errorData);
              setError('Failed to save speaking time to database');
            }

            console.log('Successfully saved speech and speaking time to database');
          } catch (dbError) {
            console.error('Error saving to database:', dbError);
            setError('Failed to save to database');
          }
        }

        // Send the accumulated speech to LLM
        await sendToLLM(speechText);
        // Clear the speech text for next turn
        setSpeechText("");
      }
      
      // Reset speaking time tracking
      totalSpeakingTimeRef.current = 0;
      speakingStartTimeRef.current = null;
      sessionStartTimeRef.current = Date.now();
      
      // Switch LLM for next response
      llmEndpointRef.current = llmEndpointRef.current === 'llm1' ? 'llm2' : 'llm1';
    }
  };

  const sendToLLM = async (text: string) => {
    try {
      // If user has raised hand or is currently speaking, don't allow LLM to speak
      if (isHandRaised || isListening || recognitionStateRef.current) {
        console.log('User has the floor, preventing LLM response');
        return;
      }
      
      // Check if another LLM is already processing
      if (llmLockRef.current) {
        console.log('Another LLM is already processing, preventing concurrent interaction');
        return;
      }
      
      // Acquire the lock
      llmLockRef.current = true;

      const endpoint = llmEndpointRef.current;
      console.log(`Attempting to connect to LLM endpoint: ${endpoint}`);
      
      // Update conversation history
      if (isHandRaised) {
        updateConversationHistory("user", text);
      } else {
        updateConversationHistory("assistant", text);
      }

      console.log('Request payload:', { 
        text, 
        topic,
        conversation_history: conversationHistoryRef.current 
      });

      const response = await fetch(`http://localhost:8080/api/${endpoint}/llm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text,
          topic,
          is_initial_message: text.includes("Let's begin the discussion"),
          is_user_message: isHandRaised,
          from_llm1: endpoint === 'llm2'
        }),
      });

      const responseText = await response.text();
      console.log('Raw response:', response);

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse response as JSON:', e);
        throw new Error(`Invalid JSON response: ${responseText}`);
      }

      if (!response.ok) {
        console.error(`LLM API error response:`, data);
        throw new Error(`LLM API error: ${response.status} - ${data.error || responseText}`);
      }
      
      if (!data.success) {
        console.error('LLM API returned unsuccessful response:', data);
        throw new Error(data.error || 'No response received from LLM');
      }

      if (!data.response) {
        console.error('No response in successful data:', data);
        throw new Error('No response received from LLM');
      }

      // Determine which participant is speaking based on the endpoint
      const speakingParticipant = endpoint === 'llm1' ? 1 : 2;
      setCurrentParticipant(speakingParticipant);

      // Notify parent component about participant speaking
      window.postMessage({
        type: 'participant_speaking',
        participantId: speakingParticipant
      }, '*');

      console.log(`Successfully received response from ${endpoint}:`, data.response);

      // Update conversation history with AI response
      updateConversationHistory("assistant", data.response);

      // Get and play the TTS audio
      try {
        console.log(`Requesting TTS from ${endpoint}`);
        const audioResponse = await fetch(`http://localhost:8080/api/${endpoint}/tts`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: data.response }),
        });
        
        if (!audioResponse.ok) {
          const errorText = await audioResponse.text();
          console.error(`TTS API error response: ${errorText}`);
          throw new Error('Failed to get TTS audio');
        }
        
        const audioBlob = await audioResponse.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Notify parent component about audio URL
        window.postMessage({
          type: 'participant_speaking',
          participantId: speakingParticipant,
          audioUrl: audioUrl
        }, '*');

        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          setIsAISpeaking(true);
          
          // Handle interruption
          const handleInterruption = () => {
            if (audioRef.current) {
              audioRef.current.pause();
              audioRef.current.currentTime = 0;
              setIsAISpeaking(false);
              setCurrentParticipant(null);
              window.postMessage({
                type: 'participant_stopped_speaking',
                participantId: speakingParticipant
              }, '*');
              if (aiSpeakingTimeoutRef.current) {
                clearTimeout(aiSpeakingTimeoutRef.current);
              }
              // Release the lock when interrupted
              llmLockRef.current = false;
            }
          };

          // Add interruption listener
          audioRef.current.addEventListener('pause', handleInterruption);
          
          audioRef.current.onended = () => {
            URL.revokeObjectURL(audioUrl);
            setIsAISpeaking(false);
            setCurrentParticipant(null);
            window.postMessage({
              type: 'participant_stopped_speaking',
              participantId: speakingParticipant
            }, '*');
            audioRef.current?.removeEventListener('pause', handleInterruption);
            
            // Release the lock when audio ends
            llmLockRef.current = false;
            
            // Add 3-second delay before next AI response
            aiSpeakingTimeoutRef.current = setTimeout(async () => {
              if (!isHandRaised && !isListening) {
                // Switch to the other LLM for the next message
                llmEndpointRef.current = llmEndpointRef.current === 'llm1' ? 'llm2' : 'llm1';
                await sendToLLM("Please continue the discussion about " + topic);
              }
            }, 3000);
          };
          
          await audioRef.current.play();
        }
      } catch (ttsError) {
        console.error('Error with TTS service:', ttsError);
        
        // Fallback to browser's speech synthesis if TTS fails
        const utterance = new SpeechSynthesisUtterance(data.response);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        // Handle interruption for browser speech synthesis
        utterance.onpause = () => {
          setIsAISpeaking(false);
          setCurrentParticipant(null);
          window.postMessage({
            type: 'participant_stopped_speaking',
            participantId: speakingParticipant
          }, '*');
          if (aiSpeakingTimeoutRef.current) {
            clearTimeout(aiSpeakingTimeoutRef.current);
          }
          // Release the lock when interrupted
          llmLockRef.current = false;
        };
        
        utterance.onend = () => {
          setIsAISpeaking(false);
          setCurrentParticipant(null);
          window.postMessage({
            type: 'participant_stopped_speaking',
            participantId: speakingParticipant
          }, '*');
          
          // Release the lock when audio ends
          llmLockRef.current = false;
          
          // Add 3-second delay before next AI response
          aiSpeakingTimeoutRef.current = setTimeout(async () => {
            if (!isHandRaised && !isListening) {
              // Switch to the other LLM for the next message
              llmEndpointRef.current = llmEndpointRef.current === 'llm1' ? 'llm2' : 'llm1';
              await sendToLLM("Please continue the discussion about " + topic);
            }
          }, 3000);
        };
        
        setIsAISpeaking(true);
        window.speechSynthesis.speak(utterance);
      }
    } catch (error) {
      console.error('Error in sendToLLM:', error);
      setError(error instanceof Error ? error.message : 'Failed to get response from LLM');
      setIsAISpeaking(false);
      setCurrentParticipant(null);
      window.postMessage({
        type: 'participant_stopped_speaking',
        participantId: currentParticipant
      }, '*');
      // Release the lock on error
      llmLockRef.current = false;
    }
  };

  // Cleanup function for timeouts
  useEffect(() => {
    return () => {
      if (aiSpeakingTimeoutRef.current) {
        clearTimeout(aiSpeakingTimeoutRef.current);
      }
      // Release the lock on component unmount
      llmLockRef.current = false;
    };
  }, []);

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-900/95 backdrop-blur-sm border-t border-gray-800 shadow-[0_-4px_20px_rgba(0,0,0,0.3)] p-6">
      <audio ref={audioRef} className="hidden" />
      <div className="max-w-4xl mx-auto">
        {error && (
          <div className="mb-4 p-3 bg-red-900/50 border border-red-800 rounded-xl flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        <div className="flex items-center gap-6">
          <button
            onClick={handleHandRaise}
            className={`p-4 rounded-full transition-all transform hover:scale-105 active:scale-95 shadow-lg ${
              isHandRaised
                ? "bg-green-600 hover:bg-green-700"
                : isAISpeaking
                ? "bg-gray-700 cursor-not-allowed"
                : "bg-gray-700 hover:bg-gray-600"
            }`}
            disabled={!canLLMsStart || isAISpeaking}
          >
            <Hand className={`w-6 h-6 ${isHandRaised ? "text-white" : "text-gray-400"}`} />
          </button>

          <button
            onClick={toggleListening}
            className={`p-4 rounded-full transition-all transform hover:scale-105 active:scale-95 shadow-lg ${
              isListening
                ? "bg-red-600 hover:bg-red-700"
                : "bg-yellow-500 hover:bg-yellow-600"
            }`}
            disabled={!isHandRaised || !canLLMsStart || isAISpeaking}
          >
            {isListening ? (
              <Mic className="w-6 h-6 text-white" />
            ) : (
              <MicOff className="w-6 h-6 text-black" />
            )}
          </button>

          <div className="flex-1 p-4 bg-gray-800 rounded-xl border border-gray-700 min-h-[60px] max-h-[100px] overflow-y-auto">
            {speechText ? (
              <p className="text-gray-300 whitespace-pre-wrap">{speechText}</p>
            ) : (
              <p className="text-gray-500 italic">
                {!canLLMsStart 
                  ? "Discussion starting in a few seconds..."
                  : isAISpeaking
                  ? "AI is speaking..."
                  : isListening
                  ? "Listening..."
                  : "Click microphone to start speaking"}
              </p>
            )}
          </div>

           
        </div>
      </div>
    </div>
  );
};

export default SpeechToText;
