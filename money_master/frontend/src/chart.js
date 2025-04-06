import React, { useState, useEffect, useContext } from "react";
import Plot from "react-plotly.js";
import { MoneyMasterContext } from "./context";
import "./style/app.css";
import "./style/chart.css";

const Chart = () => {
  const { chartData, setChartData, tickerHistory, setTickerHistory } =
    useContext(MoneyMasterContext);
  const [ticker, setTicker] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState("");

  // Fetch tickers for auto-suggestions on component mount.
  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/get-tickers");
        if (!response.ok) throw new Error("Failed to fetch tickers");
        const data = await response.json();
        if (data.tickers) {
          setSuggestions(data.tickers);
        }
      } catch (err) {
        console.error("Error fetching tickers:", err);
      }
    };
    fetchTickers();
  }, []);

  const handleChartSubmit = async (e) => {
    e.preventDefault();
    setError("");
    // Clear previous chart data in context.
    setChartData({ actualData: [], forecastData: [], lstmPrediction: null });

    try {
      const response = await fetch("http://127.0.0.1:5000/generate-chart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setChartData({
          actualData: data.actual_data || [],
          forecastData: data.forecast_data || [],
          lstmPrediction: data.lstm_prediction,
        });
        // Add the ticker to the history (if not already included)
        setTickerHistory((prevHistory) => {
          const formattedTicker = ticker.trim().toUpperCase();
          return prevHistory.includes(formattedTicker)
            ? prevHistory
            : [...prevHistory, formattedTicker];
        });
      }
    } catch (error) {
      console.error("Error generating chart:", error);
      setError("An error occurred while generating the chart.");
    }
  };

  const generateChartData = () => {
    const { actualData, forecastData } = chartData;
    if (
      !Array.isArray(actualData) ||
      !Array.isArray(forecastData) ||
      actualData.length === 0 ||
      forecastData.length === 0
    ) {
      return { data: [], layout: {} };
    }

    // Convert date strings to Date objects
    const actualDates = actualData.map((item) => new Date(item.date));
    const actualPrices = actualData.map((item) => item.close);
    const forecastDates = forecastData.map((item) => new Date(item.date));
    const forecastPrices = forecastData.map((item) => item.predicted);

    return {
      data: [
        {
          x: actualDates,
          y: actualPrices,
          type: "scatter",
          mode: "lines",
          name: "Historical Data",
          marker: { color: "blue" },
        },
        {
          x: forecastDates,
          y: forecastPrices,
          type: "scatter",
          mode: "lines",
          name: "Predicted Data (Prophet)",
          marker: { color: "red" },
        },
      ],
      layout: {
        title: "Stock Price Prediction",
        xaxis: { title: "Datetime", type: "date" },
        yaxis: { title: "Price" },
        responsive: true,
      },
    };
  };

  return (
    <div className="chart-section">
      <h2>Indian Stock Price Prediction</h2>
      <form onSubmit={handleChartSubmit}>
        <label htmlFor="ticker">Enter Stock Ticker:</label>
        <input
          type="text"
          id="ticker"
          placeholder="e.g., ZOTA"
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
        <button type="submit">Generate Chart</button>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <div id="chart">
        {chartData.actualData.length === 0 ||
        chartData.forecastData.length === 0 ? (
          <p>Chart will appear here:</p>
        ) : (
          <Plot
            data={generateChartData().data}
            layout={generateChartData().layout}
          />
        )}
      </div>
      {/* Display LSTM Next Day Prediction below the chart if available */}
      {chartData.lstmPrediction && (
        <div className="lstm-prediction">
          <h3>LSTM Next Day Prediction</h3>
          <p>Last Close: {chartData.lstmPrediction.last_close.toFixed(2)}</p>
          <p>
            Predicted Close:{" "}
            {chartData.lstmPrediction.predicted_close.toFixed(2)}
          </p>
          <p>Change: {chartData.lstmPrediction.pct_change.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
};

export default Chart;
