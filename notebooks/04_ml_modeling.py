from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("../outputs/charts", exist_ok=True)
os.makedirs("../outputs/models", exist_ok=True)

print("=" * 60)
print("MODULE 4: Machine Learning — Profit Margin Classifier")
print("=" * 60)

try:
    df = pd.read_csv("../outputs/clean_data.csv", parse_dates=["Order_Date"])
    print(f"\n✓ Loaded: {df.shape[0]:,} rows")
except FileNotFoundError:
    print("✗ Run 01_setup_and_eda.py first!")
    exit()

# ── IMPORT ML LIBRARIES ───────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score)
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    print("✓ ML libraries loaded")
except ImportError as e:
    print(f"✗ Missing library: {e}")
    print("   Run: pip install scikit-learn xgboost")
    exit()

# STEP 1: FEATURE ENGINEERING
print("\n[1] Feature Engineering...")

if "Profit_Margin" not in df.columns:
    df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100

df["Margin_Class"] = pd.cut(
    df["Profit_Margin"],
    bins=[-float("inf"), 0, 10, float("inf")],
    labels=["Low/Loss", "Medium", "High"]
)

class_counts = df["Margin_Class"].value_counts()
print("    Target class distribution:")
for cls, cnt in class_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {str(cls):<12}: {cnt:>6,} ({pct:.1f}%) {bar}")

cat_features = [col for col in ["Category", "Sub_Category", "Market", "Region", "Segment", "Ship_Mode", "Order_Priority"] if col in df.columns]
num_features = [col for col in ["Discount", "Quantity", "Shipping_Cost", "Shipping_Days", "Sales"] if col in df.columns]
time_features = [col for col in ["Year", "Month_Num", "Quarter", "Week_Num"] if col in df.columns]

all_features = cat_features + num_features + time_features
print(f"\n    Categorical features: {cat_features}")
print(f"    Numerical features : {num_features}")
print(f"    Total features : {len(all_features)}")

# STEP 2: ENCODE CATEGORICAL FEATURES
print("\n[2] Encoding categorical features...")
df_ml = df[all_features + ["Margin_Class"]].dropna().copy()
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
y = target_le.fit_transform(df_ml["Margin_Class"].astype(str))
X = df_ml[all_features].values

print(f"    Feature matrix X shape: {X.shape}")
print(f"    Classes: {list(target_le.classes_)}")

# STEP 3: TRAIN / TEST SPLIT
print("\n[3] Splitting data: 80% train, 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 4: TRAIN TUNED XGBOOST MODEL
print("\n[4] Training Tuned XGBoost Classifier...")
# Swapping out aggressive tree creation for high regularization & smooth learning steps
xgb_model = xgb.XGBClassifier(
    n_estimators=300,        
    max_depth=5,            
    learning_rate=0.05,     
    subsample=0.8,           
    colsample_bytree=0.8,
    min_child_weight=5,     # Prevents deep isolation of noisy minority samples
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)

# Smooth out the class weighting via a squareroot scale to reclaim precision balance
classes_inst, counts_inst = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
smoothed_weights = {c: np.sqrt(total_samples / (len(classes_inst) * count)) for c, count in zip(classes_inst, counts_inst)}
sample_weights = np.array([smoothed_weights[val] for val in y_train])

xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"    Test Accuracy : {xgb_accuracy:.1%}")
print(f"    F1 Score      : {xgb_f1:.3f}")
print("\n    Per-class performance:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_, digits=3))

# STEP 5: CROSS-VALIDATION
print("\n[5] 5-Fold Cross-Validation (Leakage-Free)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    
    # Re-apply matching smoothed fold weights to prevent metric distortion 
    classes_f, counts_f = np.unique(y_tr, return_counts=True)
    smooth_f = {c: np.sqrt(len(y_tr) / (len(classes_f) * count)) for c, count in zip(classes_f, counts_f)}
    fold_weights = np.array([smooth_f[val] for val in y_tr])
    
    fold_model = clone(xgb_model)
    fold_model.fit(X_tr, y_tr, sample_weight=fold_weights)
    cv_scores.append(accuracy_score(y_va, fold_model.predict(X_va)))

cv_scores = np.array(cv_scores)
print(f"    CV Accuracy scores : {[f'{s:.3f}' for s in cv_scores]}")
print(f"    Mean Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# STEP 6: COMPARISON WITH BALANCED RANDOM FOREST
print("\n[6] Comparison: Balanced Random Forest vs XGBoost...")
rf_model = RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")

print(f"    Balanced Random Forest — Accuracy: {rf_acc:.1%}  |  F1: {rf_f1:.3f}")
print(f"    Tuned XGBoost          — Accuracy: {xgb_accuracy:.1%}  |  F1: {xgb_f1:.3f}")

# STEP 7: FEATURE IMPORTANCE
print("\n[7] Feature Importance Analysis...")
importances = xgb_model.feature_importances_
feat_importance = pd.Series(importances, index=all_features).sort_values(ascending=False)
for feat, imp in feat_importance.head(5).items():
    print(f"    {feat:<20}: {imp:.4f} " + "█" * int(imp * 50))

# STEP 8: VISUALISATION
print("\n[8] Creating model result charts...")
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("ML Model Results — Profit Margin Classifier", fontsize=15, fontweight="bold")

ax1 = fig.add_subplot(gs[0, :2])   
ax2 = fig.add_subplot(gs[0, 2])    
ax3 = fig.add_subplot(gs[1, 0])   
ax4 = fig.add_subplot(gs[1, 1])   
ax5 = fig.add_subplot(gs[1, 2])   

# Plot Feature Importance
top_feats = feat_importance.head(12)
ax1.barh(top_feats.index, top_feats.values, color="#2563EB", edgecolor="white")
ax1.invert_yaxis()
ax1.set_title("Feature Importance (XGBoost)", fontweight="bold")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ax2.imshow(cm, cmap="Blues")
ax2.set_xticks(range(len(target_le.classes_)))
ax2.set_yticks(range(len(target_le.classes_)))
ax2.set_xticklabels(target_le.classes_)
ax2.set_yticklabels(target_le.classes_)
ax2.set_title("Confusion Matrix (XGBoost)", fontweight="bold")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i,j] > cm.max()/2 else "black"
        ax2.text(j, i, str(cm[i,j]), ha="center", va="center", color=color, fontweight="bold")

# Plot CV stability
ax3.bar(range(1, 6), cv_scores, color="#0D9488", alpha=0.8)
ax3.axhline(cv_scores.mean(), color="#EA580C", linestyle="--")
ax3.set_title("5-Fold CV Cross Validation", fontweight="bold")
ax3.set_ylim(0.5, 1.0)

# Plot Model Comparison
ax4.bar(["RF (Balanced)", "XGBoost"], [rf_acc, xgb_accuracy], color=["#7C3AED", "#2563EB"], alpha=0.8)
ax4.set_title("Accuracy Comparison", fontweight="bold")
ax4.set_ylim(0.5, 1.0)

# Plot Label Distributions
pred_dist = pd.Series(target_le.inverse_transform(y_pred)).value_counts()
actual_dist = pd.Series(target_le.inverse_transform(y_test)).value_counts()
ax5.bar(np.arange(3)-0.2, [actual_dist.get(c, 0) for c in target_le.classes_], 0.4, label="Actual", color="gray")
ax5.bar(np.arange(3)+0.2, [pred_dist.get(c, 0) for c in target_le.classes_], 0.4, label="Predicted", color="#16A34A")
ax5.set_xticks(range(3))
ax5.set_xticklabels(target_le.classes_)
ax5.set_title("Distribution Balance Check", fontweight="bold")
ax5.legend()

plt.savefig("../outputs/charts/11_ml_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✓ Saved: ../outputs/charts/11_ml_results.png")

# STEP 2: BUSINESS INTERPRETATION
print("\n[9] Business Interpretation...")
print("─" * 55)
print(f"  Model Global Benchmark Accuracy: {xgb_accuracy:.1%}")
print("  • Strategic Rule: Discount limits must be hard-capped at 30%.")
print("  • Risk Flagging: Orders flagged as 'Low/Loss' can be stopped pre-routing.")
print("─" * 55)

feat_importance.to_csv("../outputs/models/feature_importance.csv", header=["Importance"])
print("\n✓ Module 4 execution complete and normalized!")