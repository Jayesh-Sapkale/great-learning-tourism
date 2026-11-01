import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score

def train_and_register(data_dir=".", model_output_dir="tourism_project/deployment"):
    # Load splits from previous workflow artifact
    X_train = pd.read_csv(os.path.join(data_dir, "Xtrain.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "Xtest.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "ytrain.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "ytest.csv")).squeeze()
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    mlflow.set_experiment("Tourism_Package_Prediction")
    
    with mlflow.start_run(run_name="Gradient_Boosting_Tuned"):
        grid = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # MLflow logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(best_pipeline, "best_model")
        
        print("\n" + "=" * 60)
        print("BEST MODEL EVALUATION ON UNSEEN TEST DATA")
        print("=" * 60)
        print("Best Hyperparameters:", grid.best_params_)
        print(f"F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Save pipeline for Streamlit deployment
        os.makedirs(model_output_dir, exist_ok=True)
        model_save_path = os.path.join(model_output_dir, "best_model.joblib")
        joblib.dump(best_pipeline, model_save_path)
        print(f"Best model successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train_and_register()