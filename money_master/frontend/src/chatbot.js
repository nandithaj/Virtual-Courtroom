import React, { useState, useContext } from "react";
import { MoneyMasterContext } from "./context";
import ReactMarkdown from "react-markdown";
import "@fortawesome/fontawesome-free/css/all.min.css";
import "./style/app.css";
import "./style/chatbot.css";

const Chatbot = () => {
  const { chatHistory, setChatHistory } = useContext(MoneyMasterContext);
  const [prompt, setPrompt] = useState("");

  // Predefined instructions for fraud detection
  const predef =
    [
      "Do not use bold text",
      "Do not ever answer point by point. Only paragraphs.",
      'Check if there is a possibility of financial crime. If yes, Make the first line "POSSIBLE FRAUD DETECTED" return the crime and why.',
      "If no such possibility exists, give back responses as you normally would and ignore the previous predef",
      "ONLY content no intro. all text should be uniform(no boldening)",
      'IF a crime is detected, add a line "For more information visit" with a link leads to a website with more data ONLY IF POSSIBLE. If such a link cannot be provided, do not bother.',
    ].join(", ") + ": ";

  const handleGenerateText = async (e) => {
    e.preventDefault();

    try {
      const userMessage = { text: prompt, sender: "user" };
      const updatedHistory = [...chatHistory, userMessage];

      setChatHistory(updatedHistory);

      // Append all previous messages for context
      const fullPrompt =
        predef +
        updatedHistory
          .map(
            (msg) => `${msg.sender === "user" ? "User: " : "Bot: "}${msg.text}`
          )
          .join("\n") +
        `\nUser: ${prompt}`;

      const response = await fetch("http://127.0.0.1:5000/generate-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: fullPrompt }),
      });

      const data = await response.json();
      const botMessage = { text: data.text, sender: "bot" };

      setChatHistory((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error chatting with Gemini:", error);
      setChatHistory((prev) => [
        ...prev,
        { text: "Error contacting Gemini.", sender: "bot" },
      ]);
    }

    setPrompt("");
  };

  return (
    <div className="chatbot-section">
      <h2>Chatbot</h2>
      <div className="chat-window">
        {chatHistory.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${
              message.sender === "user" ? "user-message" : "bot-message"
            }`}
          >
            {message.sender === "bot" ? (
              <ReactMarkdown>{message.text}</ReactMarkdown>
            ) : (
              message.text
            )}
          </div>
        ))}
      </div>
      <form onSubmit={handleGenerateText} className="chat-input-form">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chatbot;
