
Deep Learning Approach to Detect Ocean Oil Spills Using SAR Images
==================================================================

This repository contains the code and report for the project titled:

"Deep Learning Approach to Detect Ocean Oil Spills Using SAR Images"

Overview
--------
This project focuses on detecting ocean oil spills using Synthetic Aperture Radar (SAR) images and deep learning models. SAR images are effective in all weather and lighting conditions, making them ideal for environmental monitoring.

The dataset used consists of 5,630 SAR images obtained from CSIRO, Australia.

Models Included
---------------
The following models were implemented and are included in this repository as Jupyter Notebook (.ipynb) files:

1. ResNet50
   - A deep convolutional neural network used for classification.
   - Processes SAR image patches for oil spill detection.

2. RCNN with VGG16 Backbone
   - Region-based CNN using VGG16 as a feature extractor.
   - Achieved the highest Precision and F1-Score in this project.

3. RCNN with InceptionV3 Backbone
   - A more complex region-based CNN using InceptionV3 for feature extraction.
   - Designed to improve generalization performance.

Report
------
The project report (PDF) is also included and contains:

- Introduction and objectives
- Dataset description and preprocessing steps
- Details of each model architecture
- Evaluation metrics and comparative results
- Conclusions and suggestions for future work

Evaluation Metrics
------------------
The following metrics were used to evaluate model performance:

- Precision
- Recall
- F1-Score

The RCNN with VGG16 backbone demonstrated the best performance across all metrics.

File Structure
--------------
- rcnn-inception-v3.ipynb
- rcnn-vgg16-final.ipynb
- resnet-50-final.ipynb
- Report.pdf
- readme.txt

Requirements
------------
To run the notebooks, the following Python libraries are recommended:

- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Use the command below to install dependencies if you have a requirements.txt file:

pip install -r requirements.txt

Future Work
-----------
- Improve segmentation on highly imbalanced SAR image patches
- Explore more advanced architectures like transformers
- Integrate explainability tools for better model transparency

License
-------
This project is intended for academic and research purposes.
