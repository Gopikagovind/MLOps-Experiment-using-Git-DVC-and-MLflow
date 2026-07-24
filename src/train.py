import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv("data/iris.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Set the MLflow experiment
mlflow.set_experiment("Iris_Experiment")

# Start MLflow run
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("Model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)

    # Log accuracy
    mlflow.log_metric("Accuracy", accuracy)

    # Save the trained model
    joblib.dump(model, "models/model.pkl")

    # Log the model file
    mlflow.log_artifact("models/model.pkl")

print("Training completed successfully!")
print("Accuracy:", accuracy)