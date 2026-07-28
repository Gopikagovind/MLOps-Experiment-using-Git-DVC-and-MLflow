# MLOps Experiment using Git, DVC and MLflow

This project demonstrates a simple Machine Learning pipeline with version control and experiment tracking using Git, DVC, and MLflow.

## Objective

Implement an ML pipeline that tracks:

- Data versioning using DVC
- Source code using Git
- Model versioning
- Experiment metrics using MLflow
- Reproducible ML workflow

## Technologies Used

- Python
- Git
- DVC (Data Version Control)
- MLflow
- Scikit-learn
- Pandas
- Joblib

## Project Structure

```
MLops_Experiment1/
│
├── data/
│   └── iris.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── prepare.py
│   └── train.py
│
├── dvc.yaml
├── dvc.lock
├── mlflow.db
└── README.md
```

## Workflow

1. Initialize Git repository.
2. Initialize DVC repository.
3. Generate the Iris dataset using `prepare.py`.
4. Track the dataset using DVC.
5. Train a Logistic Regression model using `train.py`.
6. Log parameters and accuracy using MLflow.
7. Store the trained model.
8. Reproduce the complete pipeline using DVC.

## Run the Project

Clone the repository:

```bash
git clone https://github.com/Gopikagovind/MLops-Experiment-using-Git-DVC-and-MLflow.git
```

Move into the project folder:

```bash
cd MLops-Experiment-using-Git-DVC-and-MLflow
```

Run the DVC pipeline:

```bash
dvc repro
```

Launch MLflow UI:

```bash
mlflow ui --workers 1
```

Open your browser:

```
http://127.0.0.1:5000
```

## ML Model

- Algorithm: Logistic Regression
- Dataset: Iris Dataset
- Accuracy: 1.0

## Author

**Gopika Govind**