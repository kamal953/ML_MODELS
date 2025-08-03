import seaborn as sns
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, name):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    y_pred = model.predict(X_test)
    print(f"🔹 {name}")
    print("  Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("  F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("  Training Time:", round(training_time, 4), "sec\n")


evaluate_model(DecisionTreeClassifier(max_depth=4, random_state=42), "Decision Tree")

evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")


evaluate_model(AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost")


evaluate_model(GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting")
