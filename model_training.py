import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate():
    # Load dataset
    df = pd.read_csv("features_brasileirao.csv")

    # Encode target: transforma 'result' (H, D, A) em valores numéricos
    le_result = LabelEncoder()
    y = le_result.fit_transform(df["result"])

    # Remove colunas que não devem ir para o modelo
    # X = df.drop(columns=["result", "date", "home_team", "away_team"])
    X = df.drop(columns=["result", "result_enc", "date", "home_team", "away_team"])


    # Garante que só colunas numéricas estejam presentes
    X = X.select_dtypes(include=["number"])

    # Divide dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Imputação de valores ausentes
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)

    # Modelo XGBoost
    clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Avaliação
    print("\n=== Avaliação na Validação ===")
    y_pred_val = clf.predict(X_val_imputed)
    print(classification_report(y_val, y_pred_val))

    print("\n=== Avaliação no Teste ===")
    y_pred_test = clf.predict(X_test_imputed)
    print(classification_report(y_test, y_pred_test))

    # Salvar modelo, colunas esperadas, imputer e encoder
    joblib.dump((clf, list(X.columns)), "xgb_model.pkl")
    joblib.dump(X.columns.tolist(), "expected_features.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(le_result, "label_encoder_result.pkl")
