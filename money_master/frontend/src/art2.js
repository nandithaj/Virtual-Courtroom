import React, { useState, useEffect, useContext } from "react";
import { MoneyMasterContext } from "./context";
import "./style/art2.css";

const Art2 = () => {
  const [ticker, setTicker] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState("");
  const { newsSummary, setNewsSummary, tickerHistory, setTickerHistory } =
    useContext(MoneyMasterContext);

  // Fetch tickers for auto-suggestions on component mount.
  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/get-tickers");
        if (!response.ok) throw new Error("Failed to fetch tickers");
        const data = await response.json();
        if (data.tickers) setSuggestions(data.tickers);
      } catch (err) {
        console.error("Error fetching tickers:", err);
      }
    };
    fetchTickers();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // Format ticker: trim, uppercase and append ".NS" if not present.
    let formattedTicker = ticker.trim().toUpperCase();
    if (!formattedTicker.endsWith(".NS")) {
      formattedTicker = formattedTicker + ".NS";
    }

    // If already fetched for this ticker, do nothing.
    if (newsSummary.ticker === formattedTicker && newsSummary.summary !== "") {
      return;
    }

    // Clear previous summary in context
    setNewsSummary({ ticker: formattedTicker, summary: "", sentiment: "" });

    try {
      const response = await fetch("http://127.0.0.1:5000/get-news-summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: formattedTicker }),
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setNewsSummary({
          ticker: formattedTicker,
          summary: data.summary,
          sentiment: data.sentiment,
        });
        // Add the ticker to the history (if not already included)
        setTickerHistory((prevHistory) => {
          return prevHistory.includes(formattedTicker)
            ? prevHistory
            : [...prevHistory, formattedTicker];
        });
      }
    } catch (err) {
      console.error("Error generating news summary:", err);
      setError("Error generating news summary");
    }
  };

  return (
    <div className="art2-container">
      <h2>News Summary</h2>
      <form className="art2-form" onSubmit={handleSubmit}>
        <label htmlFor="ticker">Select a Ticker:</label>
        <input
          type="text"
          id="ticker"
          placeholder="e.g., SBIN"
          list="ticker-suggestions"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
        <datalist id="ticker-suggestions">
          {suggestions
            .filter((s) => s.toLowerCase().includes(ticker.toLowerCase()))
            .map((s, index) => (
              <option key={index} value={s} />
            ))}
        </datalist>
        <button type="submit">Get News Summary</button>
      </form>
      {error && <p className="art2-error">{error}</p>}
      {newsSummary.summary && (
        <div className="art2-summary">
          <h3>Summary for {newsSummary.ticker}</h3>
          <p>{newsSummary.summary}</p>
          <h4 className={`art2-sentiment ${newsSummary.sentiment}`}>
            {newsSummary.sentiment === "green"
              ? "Positive"
              : newsSummary.sentiment === "red"
              ? "Negative"
              : "Neutral"}
          </h4>
        </div>
      )}
    </div>
  );
};

export default Art2;
