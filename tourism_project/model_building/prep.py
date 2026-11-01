import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_path="tourism_project/data/tourism.csv", output_dir="."):
    df = pd.read_csv(input_path)

    # 1. Drop non-predictive columns
    drop_cols = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    df_clean = df.drop(columns=drop_cols)

    # 2. Harmonize categorical anomalies
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].replace({'Fe Male': 'Female'})
    if 'MaritalStatus' in df_clean.columns:
        df_clean['MaritalStatus'] = df_clean['MaritalStatus'].replace({'Unmarried': 'Single'})

    # 3. Separate features and target
    X = df_clean.drop(columns=['ProdTaken'])
    y = df_clean['ProdTaken']

    # 4. Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Save splits locally as artifacts
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)

    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")
    print("Artifacts saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("=" * 60)

if __name__ == "__main__":
    prepare_data()
