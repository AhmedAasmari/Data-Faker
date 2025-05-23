# This page will explain how to fix the data from any wrong infomraiton.

## 📥 Step 1: Download the dataset.
- [fake_login_data.csv](https://github.com/user-attachments/files/19682913/fake_login_data.csv)

## 📂 Step 2: Upload to Python environment.
<img width="354" alt="image" src="https://github.com/user-attachments/assets/aabc6927-3d4a-4dd4-8025-9f0d811c051f" />

## 🐼 Step 3: Read the data using Pandas
```python
import pandas as pd
df = pd.read_csv('fake_login_data.csv')
df.head()
```
<img width="567" alt="image" src="https://github.com/user-attachments/assets/6102755d-eb7a-436d-8478-79a4d3156022" />

## 🧪 Step 4: Check columns and structure
```python
df.columns
```
<img width="636" alt="image" src="https://github.com/user-attachments/assets/a932995e-7853-48ec-a83a-17c27ac0ea61" />

- login_hour: range [0 - 23]

- login_duration: float values (e.g., 10,23)

- failed_attempts: count of failed logins

- ip_risk_score: risk score between 0.00 and 1.00

- is_attack: 0 (no attack) or 1 (attack)

## 🧼 Step 5: Check for null values
```python
df.isnull().sum()
```
- We want to make sure there are no missing (null) values in any column.
<img width="268" alt="image" src="https://github.com/user-attachments/assets/e72d1376-ddfd-4835-a7c4-c583bd81cb52" />

### ✅ The outputs are zeros, it means everything is good

## 🕒 Step 6: Check if "login_hour" contains calues outside the range 0-23
```python
print(sorted(df['login_hour'].unique()))
```
<img width="563" alt="image" src="https://github.com/user-attachments/assets/307ddf91-c39b-481f-a793-8f8ae0e936b8" />

- ✅ The result shows that all values in the column 'login_hour' are between 0 and 23.
- it means that the column is clean and does **not contain any outliers** or incorrect values.

## 📉 Step 7: Check if "login_duration" contains outliers
```python
print("Min:", df['login_duration'].min())
print("Max:", df['login_duration'].max())
```
<img width="350" alt="image" src="https://github.com/user-attachments/assets/0a4302be-0c59-4190-84a5-cb2ad244c42b" />

- 🛠️ After checking the column 'login_duration', we found a **negative value (-2.39)** which is not logically valid for session duration.
- ❗ normally, we might delete or replace such values.
- But since our project is related to **cybersecurity and instrusion detecion**, such unusual data could be a sign of **suspicious behavior**.

- ✅ Insted of removing it, we added a new column called 'is_anomaly' to flag these rows:
```python
df['is_anomaly'] = df['login_duration'] < 0
```
### And then we need to use this code
```python
df[df['is_anomaly'] == True]
```
<img width="627" alt="image" src="https://github.com/user-attachments/assets/80166829-47bb-416a-92b0-f15f4f95a5e7" />

### ✅ As shown above, the row containing a negative value in 'login_duration' has been successfully indentified and flagged using the 'is_anomaly' column.

#### This will help us later if we want to:
- filter out suspicious records
- investigate unusual login behaviors
- or use this flag as a feature in machine learning models

```python
print(df['is_anomaly'].value_counts())
```
<img width="385" alt="image" src="https://github.com/user-attachments/assets/9770c562-1cda-415b-a0a4-d1cb80264bd3" />

# 🔁 Step 8: Check the number of the attempt of the 'failed_attempts'
```python
sorted(df['failed_attempts'].unique())
```
<img width="315" alt="image" src="https://github.com/user-attachments/assets/2b0827f4-79a9-4088-bc07-28fe99555d96" />

## 📊 Step 9: Checking if the Min & Max attempts
```python
print("Min:", df['failed_attempts'].min())
print("Max:", df['failed_attempts'].max())
```
<img width="345" alt="image" src="https://github.com/user-attachments/assets/6363ad9a-1df5-4c4f-8b15-a9c82413e069" />

## ✂️ Step 10: Split the data into train and test
```python
from sklearn.model_selection import train_test_split

X = df[['login_hour', 'login_duration', 'failed_attempts', 'ip_risk_score']]
y = df['is_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- We split the dataset into training and testing sets using 70% for training and 30% for testing. This helps the AI model learn from one portion and be evaluated on unseen data to check how accurate it is.

## 🧠 Step 11: Train the AI Model
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```
- This trains the decision tree model using the training data.
We used class_weight='balanced' to help the model treat both classes fairly (attacks vs. normal logins).

## 📡 Step 12: Predict the results
```python
y_pred = model.predict(X_test)
```
- The model uses the testing data to predict whether each login is an attack or not.

## 🧾 Step 13: Evaluate the AI Model

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
- This helps us check how many correct and incorrect predictions the AI made.
We focus on *False Positives* to reduce wrong alerts.

## 📈 Step 14: Visualize Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

labels = [
    f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}",
    f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"
]
labels = np.array(labels).reshape(2, 2)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=labels, fmt='', cmap='Reds',
            xticklabels=['Predicted: Safe', 'Predicted: Attack'],
            yticklabels=['Actual: Safe', 'Actual: Attack'])
plt.title("Confusion Matrix with TP / FP / FN / TN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_result.png")  
plt.show()
```
- This chart shows how well the AI model predicted login risks.
We can clearly see False Positives, which are our target to reduce.

<img width="552" alt="image" src="https://github.com/user-attachments/assets/fd750892-b7e8-4f5d-ab07-637b6caa6a1a" />

## 🧮 Step 15: Analyze the Results
### Confusion Matrix Summary:

- **True Positives (TP):** 2 → The model correctly detected 2 attack cases.
- **True Negatives (TN):** 82 → The model correctly identified 82 safe logins.
- **False Positives (FP):** 3 → Only 3 safe logins were incorrectly predicted as attacks.
- **False Negatives (FN):** 3 → 3 real attacks were missed by the model.

🎯 **Our goal was to reduce False Positives.**
The model improved and now only 3 incorrect alerts are triggered. This is a strong result for this dataset.

## 📌 Step 16: Final Notes
### Final Notes:

- We used a decision tree classifier with `class_weight='balanced'` to handle imbalanced data.
- Data was cleaned and checked for:
  - Missing values
  - Outliers
  - Anomalies (e.g., negative durations)
- Feature selection was based on:
  - Login time
  - Duration
  - Failed attempts
  - IP risk score
- We visualized the results with a confusion matrix to track performance.

✅ This project successfully demonstrated how AI can detect risky login behavior and reduce false alerts.

🧠 We can further improve by using more advanced models like `RandomForestClassifier` or using additional features (device info, location, etc.).
