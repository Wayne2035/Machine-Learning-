🧬 Breast Cancer Case Study
📘 Overview

This project aims to classify breast cancer tumors as benign (B) or malignant (M) using machine learning algorithms.
The notebook demonstrates an end-to-end workflow including data preprocessing, feature selection, model building, and performance evaluation.

🎯 Objectives

Load and explore the Breast Cancer dataset.

Perform feature correlation analysis and remove highly correlated variables.

Standardize feature values for consistent scaling.

Train and compare three supervised learning models:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Evaluate performance using accuracy, confusion matrix, and classification reports.

📂 Dataset

The dataset breast-cancer.csv contains diagnostic measurements of cell nuclei from breast tissue samples.
Target variable:

diagnosis — M (malignant) or B (benign)

Example features include:

radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, etc.

⚙️ Workflow
1. Data Preprocessing
data = pd.read_csv("breast-cancer.csv")
x = data.drop(columns=["diagnosis", "id"])
y = data["diagnosis"]


Checked for missing and duplicate values.

Removed the id column as it doesn’t contribute to prediction.

Computed correlation matrix and dropped features with correlation > 0.85 to reduce multicollinearity.

2. Feature Scaling

Used StandardScaler from sklearn.preprocessing to normalize all features.

sc = StandardScaler()
x = sc.fit_transform(x)

3. Train-Test Split

Data was split into 75% training and 25% testing using train_test_split.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

🧠 Model Building & Evaluation
1️⃣ Logistic Regression

Achieved high accuracy and balanced precision/recall across classes.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


Accuracy: ~97%

Metrics: Excellent precision and recall for both classes.

2️⃣ K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier()
model1.fit(x_train, y_train)
y_pred_k = model1.predict(x_test)


Accuracy: ~95%

Performed slightly below Logistic Regression but still robust.

3️⃣ Decision Tree
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
y_pred_T = model2.predict(x_test)


Accuracy: ~91%

Overfitting observed compared to other models.

📊 Results Summary
Model	Accuracy	Notes
Logistic Regression	97%	Best performing model
K-Nearest Neighbors	95%	Competitive, simple
Decision Tree	91%	Slightly overfit
📈 Insights

Feature correlation removal improved model stability.

Logistic Regression provided the best generalization performance.

Scaling was essential for KNN accuracy.

Decision Tree requires pruning or parameter tuning to avoid overfitting.

🛠️ Technologies Used

Python 3.x

pandas, numpy — Data handling

matplotlib, seaborn — Visualization

scikit-learn — Model building and evaluation

Jupyter Notebook

🚀 Future Improvements

Implement Random Forest and XGBoost for better ensemble learning.

Add cross-validation and GridSearchCV for hyperparameter tuning.

Visualize ROC-AUC curves and feature importance for interpretability.

Deploy using Streamlit for interactive predictions.

👨‍💻 Author

Defeng Cao
📧 cao2020cao@gmail.com

🎓 Fordham University — M.S. in Business Analytics
🔗 LinkedIn
