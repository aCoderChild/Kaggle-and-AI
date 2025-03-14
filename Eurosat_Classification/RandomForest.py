import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data_path = "EuroSAT_13bands.csv"
df = pd.read_csv(data_path)

# Clean column names by removing unnecessary spaces
df.columns = df.columns.str.strip()

# Define feature and target columns
feature_columns = [
    "Band1", "Band2", "Band3", "Band4", "Band5", "Band6", "Band7",
    "Band8", "Band9", "Band10", "Band11", "Band12", "Band13"
]
target_column = "Label"  # Ensure this column exists in the dataset

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_columns], df[target_column], test_size=0.2,
    stratify=df[target_column], random_state=42
)

# Initialize and train the Random Forest model
rf_clf = RandomForestClassifier(n_estimators=10, random_state=9)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy Score (Test Set): {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)