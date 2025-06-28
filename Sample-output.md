# Sample Output

```
Loading and cleaning data...
Data loaded successfully. Shape: (7043, 20)
Churn rate: 26.54%

==================================================
EXPLORATORY DATA ANALYSIS
==================================================
Dataset shape: (7043, 20)
Missing values: 0

Churn Distribution:
  No: 5174 (73.46%)
  Yes: 1869 (26.54%)
Preprocessing data...

Engineering features...
Feature engineering completed. New shape: (7043, 30)
Preprocessing completed. Features: 40, Samples: 7043

Data split completed:
  Training set: 5634 samples
  Test set: 1409 samples
  Training churn rate: 0.265
  Test churn rate: 0.265
Optimizing model hyperparameters...
Fitting 5 folds for each of 320 candidates, totalling 1600 fits
Best parameters found: {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best cross-validation F1 score: 0.6204

==================================================
MODEL EVALUATION
==================================================
F1 Score: 0.6208
ROC-AUC Score: 0.8162

Detailed Classification Report:
              precision    recall  f1-score   support

       False       0.90      0.73      0.81      1035
        True       0.51      0.78      0.62       374

    accuracy                           0.75      1409
   macro avg       0.71      0.76      0.71      1409
weighted avg       0.80      0.75      0.76      1409


==============================
CROSS-VALIDATION RESULTS
==============================
F1 Scores: [0.6251298  0.63838812 0.60561915 0.60171306 0.62943072]
Mean F1: 0.6201 (±0.0141)
Mean ROC-AUC: 0.8193 (±0.0133)

==================================================
TRAINING COMPLETED SUCCESSFULLY!
==================================================

Top 10 Most Important Features:
                        feature  importance
8                IsMonthToMonth    0.699865
20  InternetService_Fiber optic    0.124223
1                        tenure    0.103823
2                MonthlyCharges    0.042268
35            Contract_Two year    0.029821
4               AvgMonthlySpend    0.000000
0                 SeniorCitizen    0.000000
3                  TotalCharges    0.000000
7             ServiceEngagement    0.000000
6                 TotalServices    0.000000

Feature importance saved to 'feature_importance.csv'
Model training completed successfully!

==================================================
READY FOR PRODUCTION!
==================================================
The model is now trained and ready to make predictions on new data.
Use trained_predictor.predict(new_data) to make predictions.
```
