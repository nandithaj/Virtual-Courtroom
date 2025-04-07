import React from "react";
import { motion } from "framer-motion";
import "../App.css";
import schrothImg from "../images/schroth.jpg";
import seasImg from "../images/seas.jpg";
import yogapilateImg from "../images/yogapilate.jpg";
import workoutImg from "../images/workout.jpg";
import bostonImg from "../images/boston.jpg";
import milwaukeeImg from "../images/milwaukee.jpg";
import charlestonImg from "../images/charleston.jpg";
import rigoImg from "../images/rigo.jpg";
import spinalfusionImg from "../images/spinalfusion.jpg";
import growingrodsImg from "../images/growingrods.jpg";
import vbtImg from "../images/vbt.jpg";

const TreatmentRecommendations = ({ angleResult }) => {
  if (!angleResult) return null;

  let recommendation = "";

  if (angleResult < 20) {
    recommendation = (
      <div className="recommendation mild">
        <h2>🟢 Treatment Recommendation - Mild Scoliosis</h2>
        <p>
          Mild scoliosis cases often do not require bracing or surgery, but
          early intervention through exercises can prevent progression.
        </p>

        <h3>Types of Corrective Exercises:</h3>
        <div className="methods">
          <div className="method">
            <img src={schrothImg} alt="Schroth Method" />
            <p>
              ✅ <b>Schroth Method</b>: Uses breathing techniques, posture
              correction, and muscle activation.
            </p>
            <a
              href="https://youtu.be/fGJEsxCYAWM?feature=shared"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={seasImg} alt="SEAS Exercise" />
            <p>
              ✅ <b>SEAS (Scientific Exercises Approach to Scoliosis)</b>:
              Personalized exercises focusing on neuromuscular control.
            </p>
            <a
              href="https://youtu.be/fex0jJtMM_M?feature=shared"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={yogapilateImg} alt="Yoga for Scoliosis" />
            <p>
              ✅ <b>Yoga & Pilates</b>: Helps improve flexibility and posture
              alignment.
            </p>
            <a
              href="https://youtu.be/vG0JdpzNw6g?feature=shared"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={workoutImg} alt="Core Strengthening" />
            <p>
              ✅ <b>Core Strengthening Workouts</b>: Helps stabilize the spine
              (e.g., planks, bird-dog exercises).
            </p>
            <a
              href="https://youtu.be/UGLMbS9INvI?feature=shared"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
        </div>
      </div>
    );
  } else if (angleResult >= 20 && angleResult <= 40) {
    recommendation = (
      <div className="recommendation moderate">
        <h2>🟡 Treatment Recommendation - Moderate Scoliosis</h2>
        <p>
          Bracing is recommended to stop further progression of the curve,
          especially in growing adolescents.
        </p>

        <h3>Types of Braces:</h3>
        <div className="methods">
          <div className="method">
            <img src={bostonImg} alt="Boston Brace" />
            <p>
              🦺 <b>Boston Brace</b>: Designed for curves in the mid to lower
              spine.
            </p>
            <a
              href="https://www.bostonoandp.com/products/scoliosis-and-spine/boston-brace/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={milwaukeeImg} alt="Milwaukee Brace" />
            <p>
              🦺 <b>Milwaukee Brace</b>: Used for high thoracic spine curves,
              includes a neck ring.
            </p>
            <a
              href="https://www.physio-pedia.com/Milwaukee_brace"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={charlestonImg} alt="Charleston Brace" />
            <p>
              🦺 <b>Charleston Bending Brace</b>: A nighttime-only brace that
              overcorrects the spine.
            </p>
            <a
              href="https://cbb.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={rigoImg} alt="Rigo Brace" />
            <p>
              🦺 <b>Rigo-Cheneau Brace</b>:A customized 3D-printed brace that
              provides better spine alignment.
            </p>
            <a
              href="https://nationalscoliosiscenter.com/rigo-cheneau-braces/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
        </div>
      </div>
    );
  } else if (angleResult > 40) {
    recommendation = (
      <div className="recommendation severe">
        <h2>🔴 Treatment Recommendation - Severe Scoliosis</h2>
        <p>
          If scoliosis progresses beyond 40°, surgical correction is often
          required.
        </p>

        <h3>Types of Scoliosis Surgeries:</h3>
        <div className="methods">
          <div className="method">
            <img src={spinalfusionImg} alt="Spinal Fusion" />
            <p>
              🔪 <b>Spinal Fusion Surgery</b>: Vertebrae are fused to prevent
              movement using rods and screws.
            </p>
            <a
              href="https://www.mayoclinic.org/tests-procedures/spinal-fusion/about/pac-20384523"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={growingrodsImg} alt="Growing Rods Surgery" />
            <p>
              🔪 <b>Growing Rods Surgery</b>: Used for children still growing,
              where rods are adjusted periodically.
            </p>
            <a
              href="https://www.spinesurgeon.in/what-are-growing-rods-what-is-their-application-in-spine-surgery-and-orthopaedics-in-general/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
          <div className="method">
            <img src={vbtImg} alt="Vertebral Body Tethering (VBT)" />
            <p>
              🔪 <b>Vertebral Body Tethering (VBT)</b>: Minimally invasive
              surgery that preserves flexibility.
            </p>
            <a
              href="https://www.chop.edu/treatments/vertebral-body-tethering-vbt"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn More
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="flex flex-col items-center p-4 border rounded-lg shadow-lg bg-white"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0, transition: { duration: 0.6 } }}
    >
      <h2 className="text-xl font-bold text-blue-600">
        Treatment Recommendation
      </h2>
      <p className="mt-2 text-lg">Cobb Angle: {angleResult}°</p>
      <p className="mt-4 text-md text-gray-700 text-center">{recommendation}</p>
    </motion.div>
  );
};

export default TreatmentRecommendations;
