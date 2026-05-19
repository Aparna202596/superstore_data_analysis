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
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score)
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    print("✓ ML libraries loaded")
except ImportError as e:
    print(f"✗ Missing library: {e}")
    print("  Run: pip install scikit-learn xgboost")
    exit()

# STEP 1: FEATURE ENGINEERING

print("\n[1] Feature Engineering...")

# Compute profit margin if not already there
if "Profit_Margin" not in df.columns:
    df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100

df["Margin_Class"] = pd.cut(
    df["Profit_Margin"],
    bins=[-float("inf"), 0, 10, float("inf")],
    labels=["Low/Loss", "Medium", "High"]
)

# Show class distribution
class_counts = df["Margin_Class"].value_counts()
print("    Target class distribution:")
for cls, cnt in class_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {str(cls):<12}: {cnt:>6,} ({pct:.1f}%) {bar}")

cat_features = []
for col in ["Category", "Sub_Category", "Sub-Category", "Market", "Region",
            "Segment", "Ship_Mode", "Order_Priority"]:
    if col in df.columns:
        cat_features.append(col)

# Numerical columns 
num_features = []
for col in ["Discount", "Quantity", "Shipping_Cost", "Shipping_Days", "Sales"]:
    if col in df.columns:
        num_features.append(col)

# Time features
time_features = []
for col in ["Year", "Month_Num", "Quarter", "Week_Num"]:
    if col in df.columns:
        time_features.append(col)

all_features = cat_features + num_features + time_features
print(f"\n    Categorical features: {cat_features}")
print(f"    Numerical features : {num_features}")
print(f"    Time features : {time_features}")
print(f"    Total features : {len(all_features)}")


# STEP 2: ENCODE CATEGORICAL FEATURES

print("\n[2] Encoding categorical features...")

df_ml = df[all_features + ["Margin_Class"]].dropna().copy()

# Store encoders so we can reverse them later
encoders = {}

for col in cat_features:
    if col in df_ml.columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
        print(f"    ✓ {col}: {len(le.classes_)} unique values → 0 to {len(le.classes_)-1}")

# Encode the target variable
target_le = LabelEncoder()
y = target_le.fit_transform(df_ml["Margin_Class"].astype(str))
X = df_ml[all_features].values

print(f"\n    Feature matrix X shape: {X.shape}   (rows × features)")
print(f"    Target vector  y shape: {y.shape}")
print(f"    Classes: {list(target_le.classes_)}")

# STEP 3: TRAIN / TEST SPLIT
print("\n[3] Splitting data: 80% train, 20% test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set : {X_train.shape[0]:,} samples")
print(f"    Test set : {X_test.shape[0]:,} samples")

# STEP 4: TRAIN XGBOOST MODEL

print("\n[4] Training XGBoost Classifier...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,        
    max_depth=5,            
    learning_rate=0.1,     
    subsample=0.8,           
    colsample_bytree=0.8,   
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0
)

xgb_model.fit(X_train, y_train)
print("    ✓ XGBoost model trained")

# Predict on test set
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\n    Test Accuracy : {accuracy:.1%}")
print(f"    F1 Score : {f1:.3f}")

# Detailed classification report
print("\n    Per-class performance:")
print(classification_report(y_test, y_pred,
                            target_names=target_le.classes_,
                            digits=3))

# STEP 5: CROSS-VALIDATION

print("\n[5] 5-Fold Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy")

print(f"    CV Accuracy scores : {[f'{s:.3f}' for s in cv_scores]}")
print(f"    Mean Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# A consistent model has low std. High std = unreliable.
if cv_scores.std() < 0.02:
    print("    ✓ Very consistent (low variance across folds)")
elif cv_scores.std() < 0.05:
    print("    ✓ Reasonably consistent")
else:
    print("    ⚠ High variance — consider more data or simpler model")

# STEP 6: COMPARE WITH RANDOM FOREST

print("\n[6] Comparison: Random Forest vs XGBoost...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")

print(f"    Random Forest — Accuracy: {rf_acc:.1%}  |  F1: {rf_f1:.3f}")
print(f"    XGBoost       — Accuracy: {accuracy:.1%}  |  F1: {f1:.3f}")
winner = "XGBoost" if accuracy >= rf_acc else "Random Forest"
print(f"    Winner: {winner}")

# STEP 7: FEATURE IMPORTANCE

print("\n[7] Feature Importance Analysis...")

importances = xgb_model.feature_importances_
feat_importance = pd.Series(importances, index=all_features).sort_values(ascending=False)

print("    Top 10 most important features:")
for feat, imp in feat_importance.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"    {feat:<20}: {imp:.4f} {bar}")

# STEP 8: VISUALISATION — MODEL RESULTS

print("\n[8] Creating model result charts...")

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("ML Model Results — Profit Margin Classifier", fontsize=15, fontweight="bold")

ax1 = fig.add_subplot(gs[0, :2])   
ax2 = fig.add_subplot(gs[0, 2])    
ax3 = fig.add_subplot(gs[1, 0])   
ax4 = fig.add_subplot(gs[1, 1])   
ax5 = fig.add_subplot(gs[1, 2])   

top_n = 12
top_feats = feat_importance.head(top_n)
colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
bars = ax1.barh(range(top_n), top_feats.values, color=colors_imp, edgecolor="white")
ax1.set_yticks(range(top_n))
ax1.set_yticklabels(top_feats.index, fontsize=10)
ax1.set_title("Feature Importance (XGBoost)", fontweight="bold")
ax1.set_xlabel("Importance Score")
for bar_obj, val in zip(bars, top_feats.values):
    ax1.text(val + 0.002, bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks(range(len(target_le.classes_)))
ax2.set_yticks(range(len(target_le.classes_)))
ax2.set_xticklabels(target_le.classes_, rotation=30, fontsize=9)
ax2.set_yticklabels(target_le.classes_, fontsize=9)
ax2.set_title("Confusion Matrix\n(diagonal = correct)", fontweight="bold")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i,j] > cm.max() / 2 else "black"
        ax2.text(j, i, str(cm[i,j]), ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

# --- Cross-Validation Scores ---
ax3.bar(range(1, 6), cv_scores, color="#2563EB", alpha=0.8, edgecolor="white")
ax3.axhline(cv_scores.mean(), color="#EA580C", linestyle="--", linewidth=2, label=f"Mean={cv_scores.mean():.3f}")
ax3.set_title("5-Fold Cross-Validation\nAccuracy Scores", fontweight="bold")
ax3.set_xlabel("Fold")
ax3.set_ylabel("Accuracy")
ax3.set_xticks(range(1, 6))
ax3.set_ylim(0.7, 1.0)
ax3.legend(fontsize=9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# --- Model Comparison ---
models = ["Random Forest", "XGBoost"]
accs = [rf_acc, accuracy]
f1s = [rf_f1, f1]
x_pos = np.arange(len(models))
w = 0.35
ax4.bar(x_pos - w/2, accs, w, label="Accuracy", color="#2563EB", alpha=0.8)
ax4.bar(x_pos + w/2, f1s,  w, label="F1 Score",  color="#16A34A", alpha=0.8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, fontsize=9)
ax4.set_ylim(0.7, 1.0)
ax4.set_title("Model Comparison", fontweight="bold")
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# --- Prediction Distribution ---
pred_classes = target_le.inverse_transform(y_pred)
actual_classes = target_le.inverse_transform(y_test)
pred_dist   = pd.Series(pred_classes).value_counts()
actual_dist = pd.Series(actual_classes).value_counts()
x_labels = pred_dist.index.tolist()
x_pos2   = np.arange(len(x_labels))
ax5.bar(x_pos2 - w/2, [actual_dist.get(l, 0) for l in x_labels], w,
        label="Actual", color="#7C3AED", alpha=0.8)
ax5.bar(x_pos2 + w/2, [pred_dist.get(l, 0) for l in x_labels], w,
        label="Predicted", color="#0D9488", alpha=0.8)
ax5.set_xticks(x_pos2)
ax5.set_xticklabels(x_labels, fontsize=9)
ax5.set_title("Actual vs Predicted\nClass Distribution", fontweight="bold")
ax5.legend(fontsize=9)

plt.savefig("../outputs/charts/11_ml_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✓ Saved: ../outputs/charts/11_ml_results.png")

# STEP 9: BUSINESS INTERPRETATION

print("\n[9] Business Interpretation...")
print("─" * 55)
print("\n  WHAT THE MODEL TELLS US:")
print()

top_feat = feat_importance.index[0]
print(f"  1. '{top_feat}' is the #1 driver of profit margin.")
print(f"     → Before approving any discount, check predicted margin class.")
print()
print(f"  2. The model is {accuracy:.0%} accurate at classifying orders.")
print(f"     → In production: flag 'Low/Loss' predictions before shipping.")
print()
print(f"  3. Key strategy insight from data:")
disc_col = "Discount" if "Discount" in df.columns else None
if disc_col:
    high_disc = df[df["Discount"] > 0.3]["Profit_Margin"].mean()
    low_disc  = df[df["Discount"] <= 0.3]["Profit_Margin"].mean()
    print(f"     Avg margin with discount >30%: {high_disc:.1f}%")
    print(f"     Avg margin with discount ≤30%: {low_disc:.1f}%")
    print(f"     → Never discount more than 30% (policy recommendation)")

print()
print("─" * 55)

# Save feature importance to CSV
feat_importance.to_csv("../outputs/models/feature_importance.csv", header=["Importance"])
print("\n✓ Saved: ../outputs/models/feature_importance.csv")

print("\n" + "=" * 60)
print("✓ MODULE 4 COMPLETE — Machine Learning Model Trained and Evaluated")
print("=" * 60)