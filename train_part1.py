import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def load_and_prepare_data(filepath):
    df = pd.read_excel(filepath)

    # Drop irrelevant columns
    df.drop(columns=['Low_Flow_Oxygen', 'High_Flow_Oxygen','Oxygen_Duration'], inplace=True)

    # Drop rows with missing target
    df = df.dropna(subset=['Any_Oxygen'])

    # Separate input and target
    X = df.drop(columns=['Any_Oxygen'])
    y = df['Any_Oxygen']

    # Identify and drop datetime columns
    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    X = X.drop(columns=datetime_cols)

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing values
    X.fillna(X.mean(), inplace=True)

    return X, y

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_and_evaluate_models(X, y):
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            import joblib

            # Assuming your SVM model is named 'model' and you're inside the loop
            if name == "SVM":
              joblib.dump(model, "svm_model.pkl")
              joblib.dump(scaler, "scaler.pkl")
              joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")
              print("SVM model saved as 'svm_model.pkl'")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            plt.show()


            # Classification Report
            print(f"\n{name}")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(classification_report(y_test, y_pred))

            report = classification_report(y_test, y_pred, output_dict=True)
            macro = report['macro avg']
            weighted = report['weighted avg']
            print("Overall Metrics:")
            print(f"Macro Avg - Precision: {macro['precision']:.3f}, Recall: {macro['recall']:.3f}, F1-score: {macro['f1-score']:.3f}")
            print(f"Weighted Avg - Precision: {weighted['precision']:.3f}, Recall: {weighted['recall']:.3f}, F1-score: {weighted['f1-score']:.3f}")

            # AUC
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                print(f"AUC Score: {auc:.3f}")
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test_scaled)
                auc = roc_auc_score(y_test, y_proba)
                print(f"AUC Score: {auc:.3f}")
            else:
                print("AUC Score: Not available for this model")

        except Exception as e:
            print(f"\n{name} failed: {e}")
              # SHAP Analysis
        try:
            print(f"\nSHAP Summary for {name}")

            if name in ["Random Forest", "XGBoost"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
            else:
                # Use a subset for KernelExplainer to reduce computation
                background = X_train_scaled[:100]
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_test_scaled[:100])
                shap_array = np.abs(shap_values[1])
                mean_shap = shap_array.mean(axis=0)
                feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Mean_SHAP_Value': mean_shap
                }).sort_values(by='Mean_SHAP_Value', ascending=False)
            # Plot SHAP summary
            top5 = feature_importance.head(5)
            plt.figure(figsize=(8, 4))
            plt.barh(top5['Feature'][::-1], top5['Mean_SHAP_Value'][::-1], color='skyblue')
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"Top 5 SHAP Features-{name}")
            plt.tight_layout()
            plt.show()

        except Exception as shap_error:
            print(f"SHAP failed for {name}: {shap_error}")




def main():
    filepath = "path/to/dataset"
    X, y = load_and_prepare_data(filepath)
    train_and_evaluate_models(X, y)

if __name__ == "__main__":
    main()
