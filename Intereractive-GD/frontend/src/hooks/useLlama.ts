import { useState } from 'react';

export const useLlama = () => {
  const [isLoading, setIsLoading] = useState(false);

  const generateResponse = async (prompt: string): Promise<string> => {
    setIsLoading(true);
    try {
      // Replace this with your actual Llama API integration
      const response = await fetch('your-llama-endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error('Error generating response:', error);
      return 'I apologize, I cannot respond at the moment.';
    } finally {
      setIsLoading(false);
    }
  };

  return { generateResponse, isLoading };
}; 