# This page will explain how to fix the data from any wrong infomraiton.

## 📥 Step 1: Download the dataset.
- [fake_login_data.csv](https://github.com/user-attachments/files/19682913/fake_login_data.csv)

## 📂 Step 2: Upload to Python environment.
<img width="354" alt="image" src="https://github.com/user-attachments/assets/aabc6927-3d4a-4dd4-8025-9f0d811c051f" />

## 🐼 Step 3: Read the data using Pandas
```
import pandas as pd
df = pd.read_csv('fake_login_data.csv')
df.head()
```
<img width="567" alt="image" src="https://github.com/user-attachments/assets/6102755d-eb7a-436d-8478-79a4d3156022" />

## 🧪 Step 4: Check columns and structure
```
df.columns
```
<img width="636" alt="image" src="https://github.com/user-attachments/assets/a932995e-7853-48ec-a83a-17c27ac0ea61" />

- login_hour: range [0 - 23]

- login_duration: float values (e.g., 10,23)

- failed_attempts: count of failed logins

- ip_risk_score: risk score between 0.00 and 1.00

- is_attack: 0 (no attack) or 1 (attack)

## Step 5: Check for null values
```
df.isnull().sum()
```
- We want to make sure there are no missing (null) values in any column.
<img width="268" alt="image" src="https://github.com/user-attachments/assets/e72d1376-ddfd-4835-a7c4-c583bd81cb52" />

### ✅ The outputs are zeros, it means everything is good

## Step 6: Check if "login_hour" contains calues outside the range 0-23
```
print(sorted(df['login_hour'].unique()))
```
<img width="563" alt="image" src="https://github.com/user-attachments/assets/307ddf91-c39b-481f-a793-8f8ae0e936b8" />

- ✅ The result shows that all values in the column 'login_hour' are between 0 and 23.
- it means that the column is clean and does **not contain any outliers** or incorrect values.

## Step 7: Check if "login_duration" contains outliers
```
print("Min:", df['login_duration'].min())
print("Max:", df['login_duration'].max())
```
<img width="350" alt="image" src="https://github.com/user-attachments/assets/0a4302be-0c59-4190-84a5-cb2ad244c42b" />

- 🛠️ After checking the column 'login_duration', we found a **negative value (-2.39)** which is not logically valid for session duration.
- ❗ normally, we might delete or replace such values.
- But since our project is related to **cybersecurity and instrusion detecion**, such unusual data could be a sign of **suspicious behavior**.

- ✅ Insted of removing it, we added a new column called 'is_anomaly' to flag these rows:
```
df['is_anomaly'] = df['login_duration'] < 0
```
### And then we need to use this code
```
df[df['is_anomaly'] == True]
```
<img width="627" alt="image" src="https://github.com/user-attachments/assets/80166829-47bb-416a-92b0-f15f4f95a5e7" />

### ✅ As shown above, the row containing a negative value in 'login_duration' has been successfully indentified and flagged using the 'is_anomaly' column.

#### This will help us later if we want to:
- filter out suspicious records
- investigate unusual login behaviors
- or use this flag as a feature in machine learning models

```
print(df['is_anomaly'].value_counts())
```
<img width="385" alt="image" src="https://github.com/user-attachments/assets/9770c562-1cda-415b-a0a4-d1cb80264bd3" />

# Step 8: Check the number of the attempt of the 'failed_attempts'
```
sorted(df['failed_attempt'].unique())
```
<img width="315" alt="image" src="https://github.com/user-attachments/assets/2b0827f4-79a9-4088-bc07-28fe99555d96" />

## Checking if the Min & Max attempts
```
print("Min:", df['failed_attempts'].min())
print("Max:", df['failed_attempts'].max())
```
<img width="345" alt="image" src="https://github.com/user-attachments/assets/6363ad9a-1df5-4c4f-8b15-a9c82413e069" />
