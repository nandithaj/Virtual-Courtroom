import React, { useEffect, useState } from 'react';
import { useLlama } from '../hooks/useLlama'; // You'll need to create this

interface ParticipantResponseProps {
  participantId: number;
  userSpeech: string;
  topic: string;
}

const ParticipantResponse: React.FC<ParticipantResponseProps> = ({
  participantId,
  userSpeech,
  topic,
}) => {
  const [response, setResponse] = useState<string>('');
  const { generateResponse } = useLlama();

  const speakResponse = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    // Use different voices for different participants
    utterance.voice = voices[participantId % voices.length] || voices[0];
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    const generateAndSpeak = async () => {
      if (userSpeech) {
        const prompt = `As participant ${participantId}, give a brief response to this point about ${topic}: "${userSpeech}"`;
        const llamaResponse = await generateResponse(prompt);
        setResponse(llamaResponse);
        speakResponse(llamaResponse);
      }
    };

    generateAndSpeak();
  }, [userSpeech, participantId, topic]);

  return (
    <div className="absolute bottom-3 right-3 bg-black/50 px-3 py-1 rounded-full">
      <p className="text-white text-sm">{response}</p>
    </div>
  );
};

export default ParticipantResponse; 