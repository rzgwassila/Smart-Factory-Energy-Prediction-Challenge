# Energy Consumption Prediction

This project aims to predict **equipment energy consumption** using machine learning models based on various environmental and time-related features. The dataset contains features such as temperature, humidity, and lighting energy, along with time-based features like hour and day.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The objective of this project is to build a predictive model that can estimate equipment energy consumption based on a variety of features, including:

- Environmental conditions (e.g., temperatures from different zones, humidity, wind speed)
- Time-based features (e.g., hour of the day)
  
A Random Forest model is employed to make predictions based on the features, after cleaning and preparing the dataset.

## Installation

To get started with this project, you will need Python 3.x along with some dependencies. You can set up the environment by following these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/energy-consumption-prediction.git
    cd energy-consumption-prediction
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

The dataset contains several columns, including temperature, humidity, and time-based features like hour and day. The preprocessing steps are as follows:

1. **Handling missing values**: Missing values are handled by removing or imputing based on the context of the data.
2. **Outlier removal**: Outliers are removed using the Interquartile Range (IQR) method to ensure data quality.
3. **Feature extraction**: Time-based features such as `hour`, `day`, and `month` are extracted from the timestamp column.

### Code Snippet for Data Preprocessing
```python
# Remove outliers using IQR
def remove_all_outliers(df, numeric_cols, threshold=1.5):
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
