// App.js
import React from "react";
import { BrowserRouter as Router } from "react-router-dom";
import "./App.css";
import Home from "./Home.js";
import { MoneyMasterProvider } from "./context.js";

function App() {
  return (
    <Router>
      <MoneyMasterProvider>
        <Home />
      </MoneyMasterProvider>
    </Router>
  );
}

export default App;
