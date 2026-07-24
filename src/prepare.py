from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert it into a DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the target column
df["target"] = iris.target

# Save it as a CSV file
df.to_csv("data/iris.csv", index=False)

print("Dataset saved successfully!")