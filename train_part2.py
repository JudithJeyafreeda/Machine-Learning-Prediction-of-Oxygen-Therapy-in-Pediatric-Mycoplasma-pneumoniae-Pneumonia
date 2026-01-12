import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score

# LightGBM
import lightgbm as lgb

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def load_and_prepare_data(filepath):
    df = pd.read_excel(filepath)

    # Drop irrelevant columns
    df.drop(columns=['Low_Flow_Oxygen', 'High_Flow_Oxygen'], inplace=True)

    # Drop rows with missing target
    df = df.dropna(subset=['Flow_Oxygen'])

    # Separate input and target
    X = df.drop(columns=['Flow_Oxygen'])
    y = df['Flow_Oxygen']

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing values
    X.fillna(X.mean(), inplace=True)

    return X, y

def train_lightgbm(X_train, X_test, y_train, y_test):
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability for class 1

    print("\n LightGBM Results")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.3f}")


def train_tabnet(X_train, X_test, y_train, y_test):
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

    clf = TabNetClassifier()
    clf.fit(
        X_train_np, y_train_np,
        eval_set=[(X_test_np, y_test_np)],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128
    )
    y_pred = clf.predict(X_test_np)
    y_proba = clf.predict_proba(X_test_np)[:, 1]  # probability for class 1

    print("\n TabNet Results")
    print(f"Accuracy: {accuracy_score(y_test_np, y_pred):.3f}")
    print(classification_report(y_test_np, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test_np, y_proba):.3f}")


def main():
    filepath = "path/to/dataset"
    X, y = load_and_prepare_data(filepath)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
    # train_tabnet(X_train, X_test, y_train, y_test)  # TabNet prefers raw features

if __name__ == "__main__":
    main()
