# Analysis of Machine Learning Models for Gamma-Ray Detection

## Description
This independent research project leverages Machine Learning techniques in Python to classify cosmic gamma-rays and hadronic cosmic-ray backgrounds in telescopic data. The study utilizes various algorithms such as Support Vector Machines, Logistic Regression, Naive Bayes, K-Nearest Neighbors, and Neural Networks for the classification task. The project involves feature engineering, regularization techniques, and emphasizes data pre-processing, model optimization, and results interpretation.

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Setting up the Dataset](#setting-up-the-dataset)
- [Training, Validation, and Testing Datasets](#datasets)
- [Classification Report Explanation](#classification-report)
- [KNN](#knn)
- [Naïve Bayes](#naive-bayes)
- [SVM](#svm)
- [Logistic Regression](#logistic-regression)
- [Logistic Regression vs. Neural Network](#comparison)
- [Neural Network](#neural-network)
- [EDITED Neural Network](#edited-neural-network)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Technology Stack](#technology-stack)
- [Code Snippet](#code-snippet)

## Installation
1. Clone the GitHub repository.
2. Ensure you have Jupyter Notebook installed.
3. Navigate to the notebook directory and run `jupyter notebook` to open the notebook.

## Usage
Open the notebook in Jupyter and execute the cells in sequence. Adjust model parameters as needed for further experimentation.

## Technology Stack
- **Core Libraries**: `numpy`, `pandas`, `matplotlib`
- **Machine Learning Libraries**: `sklearn.preprocessing`, `imblearn.over_sampling`, `sklearn.neighbors`, `sklearn.metrics`, `sklearn.naive_bayes`, `sklearn.svm`, `sklearn.linear_model`

## Code Snippet
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample code to standardize data
data = pd.read_csv('data.csv')
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
```

## Contributors
- **Syed Bilal Afzal**: Primary researcher and author.
- **Jicai Pan**: Project supervisor.

## References
- [Stanford University Notes on Classification](https://web.stanford.edu/class/cs102/lectureslides/ClassificationSlides.pdf)
- [Support Vector Machines (SVM) Notes](http://compneurosci.com/wiki/images/4/4f/Support_Vector_Machines_%28SVM%29.pdf)
- [YouTube Machine Learning Playlist](https://www.youtube.com/watch?v=i_LwzRVP7bg&list=PLjBPyabXmb_7-Zv832r2xC9oKBzygrStb&index=30&t=9352s)

## Acknowledgments
Special thanks to online course materials, textbooks, Stanford University Notes, YouTube, Codecademy, public GitHub pages, and Classification ML textbooks for providing valuable resources that supported this research.

---

**Note**: The code snippet provided is a simplified example for the sake of illustration. Please refer to the notebook for detailed implementations.
