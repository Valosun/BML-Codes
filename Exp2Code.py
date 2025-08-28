import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/content/heart.csv")

# Initial data exploration
print("Dataset Head:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())  

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nDuplicate Values:")
print(df.duplicated().sum())

print("\nSize of the Dataset:")
print(df.size)

# Data visualization
plt.figure(figsize=(5, 3))
sns.countplot(x="target", data=df)
plt.title('Heart Disease Count')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Count')
plt.show()

# Pairplot for selected features
selected_columns = ["age", "chol", "target"]  # Fixed typo in "chol"
sns.pairplot(df[selected_columns], hue="target")
plt.suptitle('Feature Relationships by Heart Disease Status', y=1.02)
plt.show()
