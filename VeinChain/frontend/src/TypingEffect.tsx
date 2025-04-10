import React, { useEffect, useRef } from "react";
import Typed from "typed.js";
import { Typography } from "@mui/joy";

interface TypingTextProps {
  text: string[];
}

const TypingText: React.FC<TypingTextProps> = ({ text }) => {
  const typedRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (typedRef.current) {
      const typed = new Typed(typedRef.current, {
        strings: text,
        typeSpeed: 50,
        backSpeed: 25,
        backDelay: 1500,
        loop: true,
      });

      return () => typed.destroy();
    }
  }, [text]);

  return (
    <Typography level="h1" className="text-3xl font-bold text-white">
      <span ref={typedRef}></span>
    </Typography>
  );
};

export default TypingText;
