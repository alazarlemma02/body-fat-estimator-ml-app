# Body Fat Estimator ML App

## Overview
This project presents an end-to-end machine learning pipeline for estimating body fat percentage using anthropometric measurements. The workflow covers data exploration, outlier detection, feature selection, model comparison, and final model deployment. The final trained model is saved and ready for integration with a Streamlit web application or other deployment platforms.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Outlier Detection](#outlier-detection)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Final Model Export](#final-model-export)
- [How to Use the Model](#how-to-use-the-model)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Project Structure

```
body-fat-estimator-ml-app/
├── app.py
├── requirements.txt
├── README.md
├── bodyfat_rf_model.pkl
├── data/
│   ├── bodyfat.csv
│   └── preprocessing.ipynb
├── images/
│   ├── knee_measurement.jpg
│   └── abdomen_measurement.jpg
```

- `data/bodyfat.csv`: The dataset containing body fat and related measurements.
- `data/preprocessing.ipynb`: The main notebook with all data analysis, modeling, and export steps.
- `bodyfat_rf_model.pkl`: The final trained Random Forest model, ready for deployment.
- `README.md`: Project documentation.

## Dataset
The dataset (`bodyfat.csv`) contains various body measurements and the target variable `BodyFat` (percentage). Each row represents an individual, with features such as age, weight, height, and several body circumferences.

## Exploratory Data Analysis
- **Distribution Plots:** Visualizations (histograms, KDE plots, boxplots, Q-Q plots) are used to understand the distribution and relationships of each feature with the target.
- **Missing Values:** The dataset is checked for missing values to ensure data quality.

## Outlier Detection
- Outliers are identified using the 3-sigma rule (values more than 3 standard deviations from the mean).
- Features with outliers are listed for further inspection or cleaning.

## Feature Selection
Multiple methods are used to select the most important features for predicting body fat:
- **Correlation Analysis:** Features with the highest absolute correlation to `BodyFat` are identified.
- **Random Forest Feature Importances:** A Random Forest model is used to rank features by importance.
- **Mutual Information:** Measures the dependency between each feature and the target.

## Model Training and Evaluation
Several regression models are trained and compared using the top 5 features from both Random Forest and Mutual Information selection:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Performance is evaluated using RMSE, MAE, and R² metrics. Random Forest with its own top 5 features achieved the best results.

## Final Model Export
- The final Random Forest model is trained on the full dataset using the top 5 most important features.
- The model is saved as `bodyfat_rf_model.pkl` (pickle format) for easy loading in a Streamlit app or other Python environments.

## How to Use the Model
1. **Load the Model:**
   ```python
   import pickle
   with open('bodyfat_rf_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```
2. **Prepare Input Data:**
   - Ensure your input is a DataFrame with the same 5 features used for training.
3. **Make Predictions:**
   ```python
   predictions = model.predict(input_data)
   ```
4. **Streamlit Integration:**
   - The model can be loaded and used in a Streamlit app for interactive predictions.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- pickle (standard library)

Install requirements with:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Acknowledgements
- The dataset and project structure are inspired by common practices in data science and machine learning.
- Special thanks to the open-source community for providing the tools and libraries used in this project.
