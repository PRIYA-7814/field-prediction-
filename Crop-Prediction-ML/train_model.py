import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

print("\nFirst 5 rows:\n", data.head())

# -------------------------------
# 📊 Simple Visualization (no seaborn)
# -------------------------------

# Crop count
data['label'].value_counts().plot(kind='bar', figsize=(10,5))
plt.title("Crop Distribution")
plt.xticks(rotation=90)
plt.show()

# -------------------------------
# 🤖 ML Model
# -------------------------------

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, pred))

# -------------------------------
# 🌱 Prediction
# -------------------------------

sample = [[90, 40, 40, 22, 80, 6.5, 200]]
prediction = model.predict(sample)

print("\nRecommended Crop:", prediction[0])