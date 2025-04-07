import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import face from "../assets/face.jpg";
import face2 from "../assets/face2.jpg";
import lips from "../assets/lips.png";
import lip2 from "../assets/lips2.png";


interface AnimatedParticipantProps {
  participantId: number;
  image: string;
  isSpeaking: boolean;
  audioUrl?: string;
}

const AnimatedParticipant: React.FC<AnimatedParticipantProps> = ({
  participantId,
  image,
  isSpeaking,
  audioUrl,
}) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.play();
    }
  }, [audioUrl]);

  useEffect(() => {
    setIsAnimating(isSpeaking);
  }, [isSpeaking]);

  return (
    <div className="relative h-full w-full">
      <audio ref={audioRef} className="hidden" />
      <motion.div
        className="relative h-full w-full"
        animate={{
          scale: 1,
          rotate: 0,
        }}
      >
        {/* Highlight border */}
        <motion.div
          className="absolute inset-0 border-4 border-transparent rounded-lg"
          animate={{
            borderColor: isAnimating
              ? ["#4CAF50", "#8BC34A", "#4CAF50"]
              : "transparent",
            borderWidth: isAnimating ? [4, 6, 4] : 0,
          }}
          transition={{
            duration: 0.4,
            repeat: isAnimating ? Infinity : 0,
            ease: "easeInOut",
          }}
        />

        {/* Base participant image */}
        <img
          src={participantId === 1 ? face : face2}
          alt={`Participant ${participantId}`}
          className="w-full h-full object-cover rounded-lg"
        />

        {/* Animated lip overlay */}
        <motion.img
          src={participantId === 1 ? lips : lip2}
          alt="Lip animation"
          className={`absolute transform -translate-x-1/2 w-[34px] h-auto z-10 ${
            participantId === 1 
              ? "bottom-[73%] left-[46%]" 
              : "bottom-[76%] left-[46%]"
          }`}
          initial={{ scale: 1, translateY: 0 }}
          animate={
            isAnimating
              ? {
                  scale: [1, 1.1, 0.95, 1],
                  translateY: [0, -1, 0.5, 0],
                }
              : {
                  scale: 1,
                  translateY: 0,
                }
          }
          transition={
            isAnimating
              ? {
                  duration: 0.4,
                  repeat: Infinity,
                  ease: "easeInOut",
                }
              : {
                  duration: 0.2,
                  ease: "easeOut",
                }
          }
          style={{
            mixBlendMode: "normal",
            objectFit: "contain",
          }}
        />
      </motion.div>
    </div>
  );
};

export default AnimatedParticipant; 