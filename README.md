Breast Cancer Prediction Model

This project implements a machine learning model using a Support Vector Classifier (SVC) to predict whether breast cancer is malignant (cancerous) or benign (non-cancerous). The model was trained and evaluated using the popular Breast Cancer Wisconsin Dataset.

Features:

Model Accuracy: Achieved an accuracy score of 95% on the holdout test set and a 96% cross-validation accuracy (3-fold).

Techniques Used:

Support Vector Classifier (SVC) for prediction.
SelectKBest for feature selection.
Confusion Matrix, Precision, Recall, F1-score for model evaluation.

Performance Metrics:

Precision: 0.96

Recall (Sensitivity): 0.91

Specificity: 0.96

F1-Score: 0.93

Dataset:

The dataset contains information on various attributes of tumors, such as radius, texture, perimeter, area, and smoothness, which help classify whether the tumor is malignant or benign.

Target Variable:

1: Malignant (cancerous)
0: Benign (non-cancerous)

Features:

30 features derived from the dataset, including cell properties like texture, smoothness, and area.

Installation:

To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/SumitPanwar0007/Breast_Cancer_Prediction_model

cd breast-cancer-prediction

Install the required dependencies:

pip install -r requirements.txt

Usage:

Load the dataset (breast_cancer.csv).
Train the model using the SVC classifier.
Evaluate the model with the test set and cross-validation.
View the prediction results and performance metrics (confusion matrix, precision, recall, F1-score).
To run the model, simply execute the following command:

python breast_cancer_prediction.py

Results:

The classifier achieved an overall accuracy of 95% on the test dataset.
Cross-validation accuracy (3-fold): 96%.
The model provides insights into important metrics such as sensitivity, specificity, precision, and F1-score.
Key Observations
Sensitivity (Recall): 91% – When the disease is present, the model predicts it correctly 91 % of the time.
Specificity: 96% – When the disease is absent, the model correctly identifies it 96% of the time.
Contributing
Feel free to submit issues or pull requests to improve the project. All contributions are welcome.
