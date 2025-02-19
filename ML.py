import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target

print("Dataset Preview:")
print(df.head())

print("\nGenerating pairwise feature plots...")
sns.pairplot(df, hue='species', palette='husl')
plt.suptitle('Pairwise Relationships of Features', y=1.02)
plt.show()

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
print("\nData split into training and testing sets.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling completed.")

rf_classifier = RandomForestClassifier(n_estimators=120, random_state=7, max_depth=8)
rf_classifier.fit(X_train_scaled, y_train)
print("\nRandom Forest model training completed.")

y_pred = rf_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.3f}")

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris_data.target_names)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(6, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="coolwarm",
    xticklabels=iris_data.target_names,
    yticklabels=iris_data.target_names,
)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importances = rf_classifier.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=iris_data.feature_names, y=feature_importances, palette="viridis")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
