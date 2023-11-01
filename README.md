Analysis of Machine Learning Models for Gamma-Ray Detection
Description
This independent research project leverages Machine Learning techniques in Python to classify cosmic gamma-rays and hadronic cosmic-ray backgrounds in telescopic data. The study utilizes various algorithms such as Support Vector Machines, Logistic Regression, Naive Bayes, K-Nearest Neighbors, and Neural Networks for the classification task. The project involves feature engineering, regularization techniques, and emphasizes data pre-processing, model optimization, and results interpretation.

Table of Contents
Abstract
Introduction
Setting up the Dataset
Training, Validation, and Testing Datasets
Classification Report Explanation
KNN
Naïve Bayes
SVM
Logistic Regression
Logistic Regression vs. Neural Network
Neural Network
EDITED Neural Network
Discussion
Conclusion
Technology Stack
Code Snippet
Installation
Clone the GitHub repository.
Ensure you have Jupyter Notebook installed.
Navigate to the notebook directory and run jupyter notebook to open the notebook.
Usage
Open the notebook in Jupyter and execute the cells in sequence. Adjust model parameters as needed for further experimentation.

Technology Stack
Core Libraries: numpy, pandas, matplotlib
Machine Learning Libraries: sklearn.preprocessing, imblearn.over_sampling, sklearn.neighbors, sklearn.metrics, sklearn.naive_bayes, sklearn.svm, sklearn.linear_model
Code Snippet
python
Copy code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample code to standardize data
data = pd.read_csv('data.csv')
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
Contributors
Syed Bilal Afzal: Primary researcher and author.
Jicai Pan: Project supervisor.
References
Stanford University Notes on Classification
Support Vector Machines (SVM) Notes
YouTube Machine Learning Playlist
Acknowledgments
Special thanks to online course materials, textbooks, Stanford University Notes, YouTube, Codecademy, public GitHub pages, and Classification ML textbooks for providing valuable resources that supported this research.
