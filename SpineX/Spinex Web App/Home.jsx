import React from "react";
import "../App.css";

const Home = () => {
  return (
    <div className="box">
      <h1 className="text-[50px] head -mt-25">Welcome to Spine-X</h1>

      <p className="para pt-5 pr-10">
        Our platform provides an advanced AI-driven solution for spinal health
        assessment. Upload spinal X-rays to classify them as normal or
        scoliosis, measure the Cobb angle to determine severity, and receive
        personalized treatment recommendations. Additionally, our posture
        analysis portal allows you to upload posture images to detect conditions
        like kyphosis and lordosis. With cutting-edge technology, we aim to
        assist in early detection and informed decision-making for better spinal
        health. Take control of your health today and get started on your
        journey to better posture and well-being!
      </p>
    </div>
  );
};

export default Home;
