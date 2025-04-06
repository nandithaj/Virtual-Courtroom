import React, { createContext, useState } from "react";

export const MoneyMasterContext = createContext();

export const MoneyMasterProvider = ({ children }) => {
  const [userData, setUserData] = useState({
    name: "",
    salary: "",
    expenses: "",
  });

  const [globalNews, setGlobalNews] = useState(null); // Save general news data
  const [newsSummary, setNewsSummary] = useState({
    summary: "",
    sentiment: "",
  }); // For summarized news (Art2)
  const [chatHistory, setChatHistory] = useState([]); // Store chat history

  // New state for chart data
  const [chartData, setChartData] = useState({
    actualData: [],
    forecastData: [],
  });

  // New state to store user ticker search history
  const [tickerHistory, setTickerHistory] = useState([]);

  return (
    <MoneyMasterContext.Provider
      value={{
        userData,
        setUserData,
        globalNews,
        setGlobalNews,
        newsSummary,
        setNewsSummary,
        chatHistory,
        setChatHistory,
        chartData,
        setChartData,
        tickerHistory,
        setTickerHistory,
      }}
    >
      {children}
    </MoneyMasterContext.Provider>
  );
};
