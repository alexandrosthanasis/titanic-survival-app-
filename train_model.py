import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def train():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(app_dir, "Titanic.csv")
    xlsx_path = os.path.join(app_dir, "Titanic.xlsx")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError("Could not find Titanic.csv or Titanic.xlsx in " + app_dir)

    print("Initial records:", len(df))

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean Fare: correct data entry artifacts where fare was multiplied by 10,000
    df["Fare"] = np.where(df["Fare"] > 1000, df["Fare"] / 10000.0, df["Fare"])
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))

    # Extract Title from Name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].fillna("Rare")

    # Family size and groupings
    df["SibSp"] = df["SibSp"].fillna(0).astype(int)
    df["Parch"] = df["Parch"].fillna(0).astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Ticket groupings (from student analysis)
    ticket_counts = df["Ticket"].value_counts()
    df["TicketCount"] = df["Ticket"].map(ticket_counts).fillna(1).astype(int)
    def classify_ticket(c):
        if c == 1:
            return "Solo (1)"
        elif c == 2:
            return "Pair (2)"
        elif c <= 4:
            return "Small Group (3-4)"
        else:
            return "Large Group (5+)"
    df["TicketGroup"] = df["TicketCount"].apply(classify_ticket)

    # Impute missing Age based on (Sex, Pclass)
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))

    # Embarked port
    df["Embarked"] = df["Embarked"].fillna("S")

    # Cabin deck letter
    def extract_deck(c):
        s = str(c).strip().upper()
        if not s or s in ("NAN", "NONE"):
            return "Unknown"
        return s[0]
    df["Deck"] = df["Cabin"].apply(extract_deck)

    # Select modeling features
    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
        "Embarked", "Title", "IsAlone", "TicketGroup"
    ]
    
    X = pd.get_dummies(df[feature_cols], columns=["Pclass", "Sex", "Embarked", "Title", "TicketGroup"], drop_first=True)
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Logistic Regression (primary for explainability & coefficients)
    lr = LogisticRegression(max_iter=1500, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    lr_cv = cross_val_score(lr, X, y, cv=5).mean()

    print(f"Logistic Regression -> Test Acc: {lr_acc:.4f}, ROC-AUC: {lr_auc:.4f}, CV-5: {lr_cv:.4f}")

    # 2. Gradient Boosting (for non-linear ensemble predictions)
    gb = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_preds)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"Gradient Boosting   -> Test Acc: {gb_acc:.4f}, ROC-AUC: {gb_auc:.4f}")

    # Coefficients series for explanation
    coefficients = pd.Series(lr.coef_[0], index=X.columns).to_dict()

    # Create model output dir
    model_dir = os.path.join(app_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    bundle = {
        "lr_model": lr,
        "gb_model": gb,
        "feature_columns": list(X.columns),
        "coefficients": coefficients,
        "intercept": float(lr.intercept_[0]),
        "metrics": {
            "lr_accuracy": lr_acc,
            "lr_auc": lr_auc,
            "lr_cv": lr_cv,
            "gb_accuracy": gb_acc,
            "gb_auc": gb_auc
        },
        "df_clean": df
    }

    bundle_path = os.path.join(model_dir, "titanic_model_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    print("Saved model bundle successfully to:", bundle_path)

if __name__ == "__main__":
    train()
