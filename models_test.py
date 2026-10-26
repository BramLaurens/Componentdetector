import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Popular Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Configuration
CSV_NAME = "calculated_data.csv"
TARGET_COLUMN = "component_name"
DROP_COLUMNS = ["photo_number", "component_name"]
VAL_SAMPLES_PER_CLASS = 50  

def benchmark_models():
    # 1. Load dataset
    try:
        df = pd.read_csv(CSV_NAME)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_NAME}' was not found. Please check the path.")
        return

    print(f"Dataset loaded successfully with {len(df)} rows.\n")

    # 2. Balanced Validation Split (Same principle as before)
    val_indices = []
    for class_name, group in df.groupby(TARGET_COLUMN):
        n_samples = min(VAL_SAMPLES_PER_CLASS, len(group))
        sampled_group = group.sample(n=n_samples, random_state=42)
        val_indices.extend(sampled_group.index)

    df_val = df.loc[val_indices]
    df_train = df.drop(val_indices)

    X_train = df_train.drop(columns=DROP_COLUMNS)
    y_train = df_train[TARGET_COLUMN]
    X_val = df_val.drop(columns=DROP_COLUMNS)
    y_val = df_val[TARGET_COLUMN]

    # 3. Define the models to benchmark
    # Note: Models sensitive to feature scales (KNN, Logistic Regression, SVM) use a Pipeline with StandardScaler.
    # Tree-based models (Decision Tree, Random Forest, Gradient Boosting) don't require scaling.
    models = {
        "K-Nearest Neighbors (KNN)": Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Support Vector Machine (SVM)": Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', SVC(random_state=42))
        ])
    }

    # 4. Train and Evaluate each model
    results = []
    
    print(f"{'Model Name':<32} | {'Accuracy':<10} | {'Weighted F1':<12} | {'Time (s)':<10}")
    print("-" * 72)

    for name, model in models.items():
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict and evaluate
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Time": train_time
        })

        print(f"{name:<32} | {acc * 100:6.2f}%    | {f1:.4f}       | {train_time:.4f}s")

    print("-" * 72)
    
    # Optional: Find the best model by accuracy
    best_model = max(results, key=lambda x: x['Accuracy'])
    print(f"\nBest Performing Model on Validation Set: {best_model['Model']} ({best_model['Accuracy'] * 100:.2f}% Accuracy)")

    return results

if __name__ == "__main__":
    benchmark_models()