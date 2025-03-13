import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "failed_logins": [1, 10, 0, 30, 2, 50, 0, 5, 3, 40],
    "data_downloaded_MB": [100, 5000, 50, 8000, 200, 15000, 30, 400, 100, 12000],
    "unusual_location": [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    "malicious_activity": [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop(columns=["malicious_activity"])
y = df["malicious_activity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(df["failed_logins"], df["data_downloaded_MB"], c=df["malicious_activity"], cmap="coolwarm", edgecolors="k")
plt.xlabel("Failed Login Attempts")
plt.ylabel("Data Downloaded (MB)")
plt.title("Cyber Threat Detection")
plt.colorbar(label="0 = Normal, 1 = Suspicious")
plt.show()
