import { motion } from "framer-motion";
import "../App.css";

const Box = ({ title, description, onClick }) => {
  return (
    <motion.div
      className="bg-gray-800 p-5 rounded-3xl shadow-lg w-80 mt-8 mr-5 cursor-pointer"
      style={{
        backgroundColor: "rgba(185, 206, 238, 0.6)", // Transparent light blue
        color: "rgb(12, 20, 26)", // Dark text color
      }}
    >
      <div className="box2">
        <h2 className="box-head">{title}</h2>
        <p className="para2">{description}</p>
      </div>
    </motion.div>
  );
};

export default Box;
