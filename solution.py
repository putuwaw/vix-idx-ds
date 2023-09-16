# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('data/loan_data_2007_2014.csv', index_col=0, low_memory=False)
df.head()

# %%
df = df.dropna(thresh=len(df)*0.95, axis=1)

# %%
df.shape

# %%
df.head(2)

# %%
selected_features = ["int_rate", "installment", "emp_length", "annual_inc", "loan_status", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "revol_util", "total_rec_late_fee", "collection_recovery_fee", "last_pymnt_amnt", "collections_12_mths_ex_med"]

# %%
selected_df = df[selected_features]
selected_df.head(2)

# %%
selected_df.info()

# %%
copy_df = selected_df.copy()
for column in copy_df.columns:
    mode_value = copy_df[column].mode().iloc[0]
    copy_df[column].fillna(mode_value, inplace=True)

# %%
copy_df.head()

# %%
copy_df['emp_length'].value_counts()

# %%
copy_df['loan_status'].value_counts()

# %%
mapping = {"10+ years": 10, "9 years": 9, "8 years": 8, "7 years": 7, "6 years": 6, "5 years": 5, "4 years": 4, "3 years": 3, "2 years": 2, "1 year": 1, "< 1 year": 0}
copy_df = copy_df.replace({'emp_length': mapping})
copy_df.head(2)

# %%
mapping = {
    'Fully Paid': 1,
    'Charged Off': 0,
    'Current': 1,
    'Default': 0,
    'Late (31-120 days)': 0,
    'In Grace Period': 0,
    'Late (16-30 days)': 0,
    'Does not meet the credit policy. Status:Fully Paid': 1,
    'Does not meet the credit policy. Status:Charged Off': 0
}
copy_df = copy_df.replace({'loan_status': mapping})
copy_df.head(2)

# %%
X = copy_df.drop('loan_status', axis=1)
y = copy_df['loan_status']

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# %%
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)


# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


