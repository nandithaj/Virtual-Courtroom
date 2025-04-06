// UserForm.js
import React, { useState, useContext } from "react";
import { MoneyMasterContext } from "./context";
import "./style/form.css";

function UserForm() {
  const { setUserData } = useContext(MoneyMasterContext);
  const [formValues, setFormValues] = useState({
    name: "",
    salary: "",
    expenses: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setUserData((prev) => ({ ...prev, ...formValues }));
  };

  return (
    <div className="user-form-container">
      <div className="user-form-box">
        <h2>Get the Right Financial Help for You</h2>
        <p>
          Enter your details below to receive personalized insights and
          recommendations.
        </p>
        <form onSubmit={handleSubmit}>
          <div className="form-field">
            <label htmlFor="name">Name:</label>
            <input
              type="text"
              name="name"
              id="name"
              value={formValues.name}
              onChange={handleChange}
              placeholder="Your Name"
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="salary">Salary:</label>
            <input
              type="number"
              name="salary"
              id="salary"
              value={formValues.salary}
              onChange={handleChange}
              placeholder="Your Monthly Salary"
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="expenses">Expenses:</label>
            <input
              type="number"
              name="expenses"
              id="expenses"
              value={formValues.expenses}
              onChange={handleChange}
              placeholder="Your Monthly Expenses"
              required
            />
          </div>
          <button type="submit" className="form-submit-btn">
            Save Details
          </button>
        </form>
      </div>
    </div>
  );
}

export default UserForm;
