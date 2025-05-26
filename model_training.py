import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report

def train_and_evaluate(train_file='train.csv', val_file='val.csv', test_file='test.csv', model_file='model.pkl'):
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)

    features = [col for col in train.columns if col not in ['result', 'result_enc', 'date', 'home_team', 'away_team']]
    X_train, y_train = train[features], train['result_enc']
    X_val, y_val = val[features], val['result_enc']
    X_test, y_test = test[features], test['result_enc']

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    print("\n=== Avaliação na Validação ===")
    print(classification_report(y_val, clf.predict(X_val)))
    print("\n=== Avaliação no Teste ===")
    print(classification_report(y_test, clf.predict(X_test)))

    joblib.dump(clf, model_file)