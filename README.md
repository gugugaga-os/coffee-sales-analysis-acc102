Macroeconomic Business Cycle Risk Analysis Project

This is a full Python data science project for analyzing U.S. macroeconomic trends, constructing a recession risk score, and predicting economic slowdowns using a custom-built logistic regression model.

 Project Overview

The project takes quarterly U.S. macroeconomic data (1959Q1–2009Q3) and builds an end-to-end pipeline for:

• Data cleaning and feature engineering

• Exploratory data analysis (EDA) and visualization

• Custom risk score construction

• Handwritten logistic regression modeling

• Model evaluation and automated report generation

All analysis is reproducible and outputs a complete HTML dashboard report.

 Repository Structure

├── main.py
├── requirements.txt
├── us_macro_public_dataset.csv
└── outputs/
    ├── figures/
    ├── tables/
    └── macro_business_cycle_report.html

 Setup and Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the pipeline:

python main.py

All results (charts, tables, report) will be saved to the outputs/ folder automatically.

 Key Features

1. Data Preprocessing

◦ Time-series indexing and missing value handling

◦ Derived features: GDP growth rates, inflation-adjusted interest rates, and demand component ratios

◦ Custom Macroeconomic Risk Score (0–100) combining unemployment, inflation, and growth metrics

2. Exploratory Data Analysis

◦ Trend charts for key economic indicators

◦ Correlation heatmaps and scatter plots

◦ Visualization of business cycle risk patterns

3. Predictive Modeling

◦ Handwritten logistic regression classifier (no scikit-learn)

◦ Time-based train/test split to avoid look-ahead bias

◦ Target: Predict whether the next quarter will see an economic slowdown

◦ Evaluation metrics: accuracy, precision, recall, F1-score, AUC, and confusion matrix

◦ Permutation importance and bootstrap stability testing

4. Automated Reporting

◦ All visualizations saved as scalable SVG files

◦ Cleaned datasets and model metrics exported as CSV

◦ Interactive HTML dashboard with key findings and charts

 Main Results

• The custom risk score effectively identifies historical periods of elevated economic stress (e.g., the 1970s stagflation and 2008 crisis).

• Inflation and unemployment are the most predictive factors for economic slowdowns in this dataset.

• The logistic regression model achieves stable performance on out-of-sample data, confirming the usefulness of the engineered features.

 Notes

• The dataset ends in 2009 and does not include recent economic shocks.

• The logistic regression is a simple linear model; future work could explore more advanced algorithms or feature engineering techniques.

requirements.txt

pandas>=1.5
numpy>=1.23








    
