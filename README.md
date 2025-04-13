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

- ✅ The outputs are zeros, it means everything is good

## Step 6: Check if "login_hour" contains calues outside the range 0-23
```
print(sorted(df['login_hour'].unique()))
```
<img width="563" alt="image" src="https://github.com/user-attachments/assets/307ddf91-c39b-481f-a793-8f8ae0e936b8" />

- ✅ The result shows that all values in the column 'login_hour' are between 0 and 23.
- it means that the column is clean and does **not contain any outliers** or incorrect values.

## Step 7: Check if "login_duration" contains outliers
```
print(df['login_duration'].min())
print(df['login_duration'].max())
```
<img width="314" alt="image" src="https://github.com/user-attachments/assets/4de8f112-0883-414b-a1ec-6de334ce1958" />

- 🛠️ After checking the column 'login_duration', we found a **negative value (-2.39)** which is not logically valid for session duration.
- ❗ normally, we might delete or replace such values.
- But since our project is related to **cybersecurity and instrusion detecion**, such unusual data could be a sign of **suspicious behavior**.

- ✅ Insted of removing it, we added a new column called 'is_anomaly' to flag these rows:
```
df['is_anomaly'] = df['login_duration'] < 0
```


