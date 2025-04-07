# 🦴 Spinex: Automated Spinal & Posture Disease Detection System

**Spinex** is an intelligent diagnostic system that detects **scoliosis** and **postural deformities like Lordosis and Kyphosis** from medical images using deep learning. It offers accurate diagnosis, Cobb’s angle measurement, and treatment recommendations — all in one platform.

---

## 🔍 Modules Overview

### 📌 1. Scoliosis Detection

- **Input**: X-ray Images
- **Models**:
  - `CNN_Scoliosis_Detection.ipynb` – Custom CNN model
  - `InceptionV3_Scoliosis_Detection.ipynb` – Transfer Learning with InceptionV3
- **Post-Detection Analysis**:
  - Cobb's Angle Measurement:
    - 🧠 `Automatic_segmentation_and_Keypoint_detection.ipynb` – Auto keypoint detection
    - ✍️ `Manual_selection_for_Cobb_Angle_Measurement.ipynb` – Manual keypoint selection for precision
  - Outputs: Curve Type(C curve or S curve), Angle, Severity (Mild, Moderate or Severe)
  - 🩺 Treatment Recommendations

---

### 📌 2. Posture Deformity Detection

- **Input**: Side posture images
- **Model**:
  - `EfficientNetB0_Posture_Deformity_Detection.ipynb` – EfficientNetB0-based TL model
- **Classifies** into Normal, Lordosis or Kyphosis
- 💡 Suggests treatment based on severity

---

## ⚙️ Getting Started

Follow these steps to set up and use **Spinex**:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/Spinex.git
   cd Spinex
   ```


2. **Install dependencies**
```
pip install -r requirements.txt
```

3. **Run the appropriate notebooks to get started:**

- 🩻 **Scoliosis Detection**:  
  - [`CNN_Scoliosis_Detection.ipynb`](./CNN_Scoliosis_Detection.ipynb) 
  - [`InceptionV3_Scoliosis_Detection.ipynb`](./InceptionV3_Scoliosis_Detection.ipynb)

- 📐 **Cobb’s Angle Measurement**:  
  - [`Automatic_segmentation_and_Keypoint_detection.ipynb`](./Automatic_segmentation_and_Keypoint_detection.ipynb)
  - [`Manual_selection_for_Cobb_Angle_Measurement.ipynb`](./Manual_selection_for_Cobb_Angle_Measurement.ipynb) 

- 🧍‍♀️ **Posture Deformity Detection**:  
  - [`EfficientNetB0_Posture_Deformity_Detection.ipynb`](./EfficientNetB0_Posture_Deformity_Detection.ipynb) 

---

## 🧠 Tech Stack

- **Languages**: Python  
- **Tools**: Jupyter Notebook, OpenCV, NumPy, Matplotlib  
- **Frameworks**: TensorFlow, Keras  
- **Models**: CNN, InceptionV3, EfficientNetB0

---

## 🤝 Contributions

Feel free to **fork** this repo, **open issues**, or **submit pull requests**.  
Your contributions make **Spinex** better! 💪

