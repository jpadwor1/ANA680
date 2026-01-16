import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TEST_SIZE = 0.25

DATA_PATH = "./Server/breast-cancer-wisconsin-data1.csv" 

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head()

data = df.copy()

if "bare_nuclei" in data.columns:
    data["bare_nuclei"] = pd.to_numeric(data["bare_nuclei"].replace("?", np.nan))

if "ID" in data.columns:
    data = data.drop(columns=["ID"])

X = data.drop(columns=["class"])
y = data["class"].map({2: 0, 4: 1})  #benign->0, malignant->1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train class counts:\n", y_train.value_counts())


TOP_K = 4 

# Impute missing values on training set for correlation calculation
imp_for_corr = SimpleImputer(strategy="median")
X_train_imp_for_corr = pd.DataFrame(
    imp_for_corr.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Correlation of each feature with target
corrs = X_train_imp_for_corr.apply(lambda s: s.corr(y_train))
corrs_sorted = corrs.reindex(corrs.abs().sort_values(ascending=False).index)

selected_features = list(corrs_sorted.index[:TOP_K])

print("Correlations (sorted by abs value):")
print(pd.DataFrame({"feature": corrs_sorted.index, "corr_with_target": corrs_sorted.values}))

print(f"\nSelected top {TOP_K} features:")
selected_features

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_sel)
X_test_imp = imputer.transform(X_test_sel)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

svm_rbf = SVC(kernel="rbf", random_state=RANDOM_STATE)
svm_rbf.fit(X_train_scaled, y_train)

# Evaluate
pred = svm_rbf.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
print(f"Kernel SVM (RBF) accuracy using top {TOP_K} features: {acc:.4f}")

bundle = {
    "selected_features": selected_features,
    "label_map": {2: 0, 4: 1},
    "imputer": imputer,
    "scaler": scaler,
    "model": svm_rbf,
}

MODEL_PATH = "svm_rbf_top_features.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

print("Saved:", MODEL_PATH)

with open("svm_rbf_top_features.pkl", "rb") as f:
    loaded = pickle.load(f)

sel = loaded["selected_features"]
imputer_loaded = loaded["imputer"]
scaler_loaded = loaded["scaler"]
model_loaded = loaded["model"]

# Example: predict first 5 test rows
X_example = X_test[sel]
X_example_imp = imputer_loaded.transform(X_example)
X_example_scaled = scaler_loaded.transform(X_example_imp)

pred_example = model_loaded.predict(X_example_scaled)
print("Preds (first 5):", pred_example)
print("True  (first 5):", y_test.iloc[:5].to_numpy())
print("Done")