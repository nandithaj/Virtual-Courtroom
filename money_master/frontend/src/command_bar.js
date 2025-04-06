import React from "react";
import "@fortawesome/fontawesome-free/css/all.min.css";
import "./App.css";

function Banner({ setCont }) {
  return (
    <div className="Banner">
      <div className="icon-container">
        <i className="fas fa-home" title="Home" onClick={() => setCont(0)}></i>
        <i
          className="fas fa-solid fa-robot"
          title="Chatbot"
          onClick={() => setCont(1)}
        ></i>
        <i
          className="fas fa-sharp fa-solid fa-chart-simple"
          title="Stock Price Prediction"
          onClick={() => setCont(2)}
        ></i>
        <i
          className="fas fa-regular fa-newspaper"
          title="Article Summarization"
          onClick={() => setCont(3)}
        ></i>
      </div>
    </div>
  );
}

export default Banner;
