import json

nb = json.load(open('code.ipynb', encoding='utf-8'))

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

# ── 1. Change threshold 0.7 → 0.6 everywhere ──────────────────────────────
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if '0.7' in src:
        new_src = src.replace(
            "high_risk_df = df[df['predicted_risk_score'] > 0.7].copy()\n"
            "print(f\"High-risk records (score > 0.7): {len(high_risk_df)}\")",
            "high_risk_df = df[df['predicted_risk_score'] > 0.6].copy()\n"
            "print(f\"High-risk records (score > 0.6): {len(high_risk_df)}\")"
        )
        new_src = new_src.replace(
            "### Stage 2: Classification — Predict risk_category (only when risk_score > 0.7)",
            "### Stage 2: Classification — Predict risk_category (only when risk_score > 0.6)"
        )
        new_src = new_src.replace(
            "Stage 2: If risk_score > 0.7,",
            "Stage 2: If risk_score > 0.6,"
        )
        new_src = new_src.replace(
            "if score > 0.7:",
            "if score > 0.6:"
        )
        new_src = new_src.replace(
            "risk_score > 0.7",
            "risk_score > 0.6"
        )
        cell['source'] = [new_src]

# ── 2. Build GridSearchCV cell ─────────────────────────────────────────────
gridsearch_code = '''from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SKPipeline

print("Starting GridSearchCV — this may take a few minutes...")

RISK_THRESHOLD = 0.6

# Re-filter high-risk subset with updated threshold
high_risk_df_gs = df[df['predicted_risk_score'] > RISK_THRESHOLD].copy()
X_gs = high_risk_df_gs[FEATURES].fillna(0)
y_gs = le.transform(high_risk_df_gs['risk_category'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler_gs = RobustScaler()
X_gs_sc = scaler_gs.fit_transform(X_gs)

# ── Grid definitions ──────────────────────────────────────────────────────
param_grids = {}

param_grids['RandomForest'] = {
    'model': [RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)],
    'model__n_estimators': [200, 300],
    'model__max_depth': [8, 12, None],
    'model__min_samples_leaf': [1, 2, 4],
}

param_grids['GradientBoosting'] = {
    'model': [GradientBoostingClassifier(random_state=42)],
    'model__n_estimators': [200, 300],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0],
}

if XGB_AVAILABLE:
    param_grids['XGBoost'] = {
        'model': [XGBClassifier(random_state=42, verbosity=0, n_jobs=-1, eval_metric='mlogloss')],
        'model__n_estimators': [200, 300],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    }

if LGB_AVAILABLE:
    param_grids['LightGBM'] = {
        'model': [LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, class_weight='balanced')],
        'model__n_estimators': [200, 300],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.05, 0.1],
        'model__num_leaves': [31, 63],
    }

# ── Run GridSearchCV ──────────────────────────────────────────────────────
gs_results = {}
for name, param_grid in param_grids.items():
    pipe = SKPipeline([('model', param_grid['model'][0])])
    pg = {k: v for k, v in param_grid.items() if k != 'model'}
    gs = GridSearchCV(pipe, pg, cv=cv, scoring='f1_weighted',
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_gs_sc, y_gs)
    best_score = gs.best_score_
    gs_results[name] = {'best_score': best_score, 'best_params': gs.best_params_, 'gs': gs}
    print(f"{name:20s} | Best CV F1 (weighted): {best_score:.4f} | Params: {gs.best_params_}")

# ── Select best tuned model ───────────────────────────────────────────────
best_tuned_name = max(gs_results, key=lambda k: gs_results[k]['best_score'])
best_tuned_model = gs_results[best_tuned_name]['gs'].best_estimator_
print(f"\\n✅ Best tuned model: {best_tuned_name} | CV F1: {gs_results[best_tuned_name]['best_score']:.4f}")

# ── Evaluate on held-out test set ─────────────────────────────────────────
X_cls_test_gs = scaler_gs.transform(X_cls_test)
tuned_preds = best_tuned_model.predict(X_cls_test_gs)
tuned_acc  = accuracy_score(y_cls_test, tuned_preds)
tuned_prec = precision_score(y_cls_test, tuned_preds, average='weighted', zero_division=0)
tuned_rec  = recall_score(y_cls_test, tuned_preds, average='weighted', zero_division=0)
tuned_f1   = f1_score(y_cls_test, tuned_preds, average='weighted', zero_division=0)

print(f"\\nTuned Model Test Performance:")
print(f"  Accuracy  : {tuned_acc:.4f}")
print(f"  Precision : {tuned_prec:.4f}")
print(f"  Recall    : {tuned_rec:.4f}")
print(f"  F1-score  : {tuned_f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_cls_test, tuned_preds, target_names=le.classes_, zero_division=0))

# ── Compare baseline vs tuned ─────────────────────────────────────────────
baseline_f1 = cls_results[best_cls_name]['F1']
improvement  = tuned_f1 - baseline_f1
print(f"\\nBaseline best ({best_cls_name}) F1 : {baseline_f1:.4f}")
print(f"Tuned best ({best_tuned_name}) F1    : {tuned_f1:.4f}")
print(f"Improvement                          : {improvement:+.4f}")

# ── Confusion matrix for tuned model ─────────────────────────────────────
cm_tuned = confusion_matrix(y_cls_test, tuned_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix — Tuned {best_tuned_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ── Override best_cls_model with tuned version if it improves F1 ─────────
if tuned_f1 > baseline_f1:
    best_cls_model  = best_tuned_model
    best_cls_name   = f"{best_tuned_name} (Tuned)"
    scaler_cls      = scaler_gs
    print(f"\\n✅ Tuned model adopted as final classifier.")
else:
    print(f"\\nℹ️  Baseline model retained (tuned model did not improve F1).")
'''

gridsearch_md = (
    "> **GridSearchCV Tuning Results:** Cross-validated grid search systematically explores hyperparameter combinations "
    "to maximise weighted F1-score. The 5-fold stratified CV ensures each fold preserves class proportions. "
    "Key parameters tuned: tree depth (controls model complexity), n_estimators (ensemble size), learning rate "
    "(boosting step size), and subsample (stochastic gradient boosting regularisation). "
    "The confusion matrix of the tuned model should show improved diagonal values compared to the baseline. "
    "If the tuned F1 exceeds the baseline, the tuned model is automatically adopted as the final classifier for saving and deployment."
)

# ── 3. Insert GridSearchCV cell + its markdown AFTER cell index 37 (classification comparison markdown) ──
new_cells = []
for i, cell in enumerate(nb['cells']):
    new_cells.append(cell)
    if i == 37:   # after classification comparison markdown
        new_cells.append(code_cell(gridsearch_code))
        new_cells.append(md_cell(gridsearch_md))

nb['cells'] = new_cells

with open('code.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")

# Verify threshold changes
src_all = ' '.join(''.join(c['source']) for c in nb['cells'])
count_07 = src_all.count('> 0.7')
count_06 = src_all.count('> 0.6')
print(f"Occurrences of '> 0.7': {count_07}  (should be 0)")
print(f"Occurrences of '> 0.6': {count_06}  (should be >= 3)")
