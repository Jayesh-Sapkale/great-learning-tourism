import os
import sys
import pandas as pd

EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

def register_data(file_path="tourism_project/data/tourism.csv"):
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print("=" * 60)
    print("DATA REGISTRATION & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Dataset path: {file_path}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Validation FAILED: Missing expected columns: {missing_cols}")
        sys.exit(1)
    else:
        print("Schema Validation: PASSED (All expected columns present).")

    print("\nTarget ('ProdTaken') Class Distribution:")
    print(df['ProdTaken'].value_counts(normalize=True).mul(100).round(2).to_string())
    print("\nMissing Values Summary:")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().any() else "No missing values.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    register_data()
