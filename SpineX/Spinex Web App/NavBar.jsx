import React from "react";
import "../App.css";

const Navbar = ({ onNavigate }) => {
  return (
    <nav className="p-4 navbar">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-white text-1xl font-bold">SpineX</h1>
        <ul className="flex space-x-6">
          <li>
            <a
              href="#"
              className="text-white hover:text-gray-300 text-[14px]"
              onClick={(e) => {
                e.preventDefault(); // Prevent default link behavior
                onNavigate(); // Call function to go to Home
              }}
            >
              Home
            </a>
          </li>
          <li>
            <a href="#" className="text-white hover:text-gray-300 text-[14px]">
              About Us
            </a>
          </li>
        </ul>
        <div>
          <a href="#" className="text-white hover:text-gray-300 text-[14px]">
            Login
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
