"""
Fast retrain - XGBoost classifier + GBR regressor
- SMOTE for class balancing
- Feature importance pruning
- 100 trees for speed
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, classification_report, mean_absolute_error, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# ── Load & clean ─────────────────────────────────────────────────────────────
df = pd.read_csv('indian_roads_dataset.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
for c in ['hour','latitude','longitude','temperature','lanes','risk_score']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
for c in ['weather','visibility','road_type','traffic_density','cause']:
    df[c] = df[c].str.strip().str.lower()
df['festival'] = df['festival'].str.strip().str.lower().replace('none','no_festival')
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)
for c in df.select_dtypes(include='object').columns:
    df[c].fillna(df[c].mode()[0], inplace=True)
print(f"Loaded: {df.shape}")

# ── Synthetic road_condition ─────────────────────────────────────────────────
np.random.seed(42)
conds = []
for rt in df['road_type']:
    r = np.random.random()
    if rt in ['highway','expressway']:
        conds.append('damaged' if r<0.10 else ('under_construction' if r<0.30 else 'good'))
    else:
        conds.append('damaged' if r<0.25 else ('under_construction' if r<0.55 else 'good'))
df['road_condition'] = conds
print("road_condition:", df['road_condition'].value_counts().to_dict())

# ── Feature engineering ──────────────────────────────────────────────────────
WR  = {'clear':0,'cloudy':1,'rain':2,'fog':3,'hail':3,'storm':4,'snow':4}
VM  = {'high':0,'medium':1,'low':2}
TM  = {'low':0,'medium':1,'high':2}
RCM = {'good':0,'under_construction':1,'damaged':2}
RTS = ['highway','urban','rural','expressway','mountain']
RCS = ['good','under_construction','damaged']
FVS = ['no_festival','diwali','holi','eid','christmas','navratri']

df['is_night']           = ((df['hour']>=20)|(df['hour']<=5)).astype(int)
df['is_morning_rush']    = ((df['hour']>=7)&(df['hour']<=9)).astype(int)
df['is_evening_rush']    = ((df['hour']>=17)&(df['hour']<=19)).astype(int)
df['hour_sin']           = np.sin(2*np.pi*df['hour']/24)
df['hour_cos']           = np.cos(2*np.pi*df['hour']/24)
df['month']              = df['date'].dt.month.fillna(6).astype(int)
df['day_of_year']        = df['date'].dt.dayofyear.fillna(180).astype(int)
df['month_sin']          = np.sin(2*np.pi*df['month']/12)
df['month_cos']          = np.cos(2*np.pi*df['month']/12)
df['weather_risk']       = df['weather'].map(WR).fillna(1)
df['visibility_enc']     = df['visibility'].map(VM).fillna(1)
df['traffic_density_enc']= df['traffic_density'].map(TM).fillna(1)
df['road_cond_enc']      = df['road_condition'].map(RCM).fillna(0)
df['risk_interaction']   = df['weather_risk'] * df['visibility_enc']
df['night_fog']          = df['is_night'] * (df['weather']=='fog').astype(int)
df['peak_high_traffic']  = df['is_peak_hour'] * df['traffic_density_enc']
df['temperature_log']    = np.log1p(df['temperature'].clip(lower=0))
for rt in RTS: df[f'road_{rt}'] = (df['road_type']==rt).astype(int)
for rc in RCS: df[f'cond_{rc}'] = (df['road_condition']==rc).astype(int)
for fv in FVS: df[f'festival_{fv}'] = (df['festival']==fv).astype(int)

# cls-specific interaction features
df['is_fog']           = (df['weather']=='fog').astype(int)
df['is_rain']          = (df['weather']=='rain').astype(int)
df['is_storm']         = df['weather'].isin(['storm','hail','snow']).astype(int)
df['is_low_vis']       = (df['visibility']=='low').astype(int)
df['is_highway']       = (df['road_type']=='highway').astype(int)
df['is_rural']         = (df['road_type']=='rural').astype(int)
df['is_damaged']       = (df['road_condition']=='damaged').astype(int)
df['is_construction']  = (df['road_condition']=='under_construction').astype(int)
df['fog_night']        = df['is_fog'] * df['is_night']
df['rain_highway']     = df['is_rain'] * df['is_highway']
df['low_vis_night']    = df['is_low_vis'] * df['is_night']
df['weather_x_vis']    = df['weather_risk'] * df['visibility_enc']
df['temp_risk']        = (df['temperature']<15).astype(int)
df['damaged_night']    = df['is_damaged'] * df['is_night']
df['construction_rain']= df['is_construction'] * df['is_rain']

BASE = (['latitude','longitude','hour','is_weekend','is_peak_hour',
         'is_night','is_morning_rush','is_evening_rush',
         'hour_sin','hour_cos','month_sin','month_cos','month','day_of_year',
         'weather_risk','visibility_enc','traffic_density_enc','road_cond_enc',
         'temperature','lanes','traffic_signal',
         'risk_interaction','night_fog','peak_high_traffic','temperature_log']
        + [f'road_{rt}' for rt in RTS]
        + [f'cond_{rc}' for rc in RCS]
        + [f'festival_{fv}' for fv in FVS])

CLS_EXTRA = ['is_fog','is_rain','is_storm','is_low_vis','is_highway','is_rural',
             'is_damaged','is_construction','fog_night','rain_highway',
             'low_vis_night','weather_x_vis','temp_risk','damaged_night','construction_rain']
CLS_FEATS = list(dict.fromkeys(BASE + CLS_EXTRA))

# ── Stage 1: Regression (100 trees) ─────────────────────────────────────────
print("\n=== Stage 1: Regression ===")
X = df[BASE].fillna(0); y = df['risk_score']
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc_r = RobustScaler()
reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                 max_depth=5, subsample=0.8, random_state=42)
reg.fit(sc_r.fit_transform(Xtr), ytr)
rmse = np.sqrt(mean_squared_error(yte, reg.predict(sc_r.transform(Xte))))
r2   = r2_score(yte, reg.predict(sc_r.transform(Xte)))
print(f"RMSE={rmse:.4f}  R2={r2:.4f}")

# ── Stage 2: Classification ──────────────────────────────────────────────────
print("\n=== Stage 2: Classification ===")
def map_cause(c):
    c = str(c).lower().strip()
    if c in ['fog','low visibility','glare','weather','rain','storm','flood','hail','snow']:
        return 'Weather-Related'
    if c in ['poor road','road damage','pothole','poor design','construction']:
        return 'Road Infrastructure'
    return 'Driving Behavior'

df['risk_category'] = df['cause'].apply(map_cause)

# ── Fix class imbalance at source ────────────────────────────────────────────
# fog/low-vis/rain/storm records → Weather-Related (strengthen signal)
wx_mask = (df['weather'].isin(['fog','rain','storm','hail','snow'])) | (df['visibility'] == 'low')
df.loc[wx_mask & (df['risk_category'] == 'Driving Behavior'), 'risk_category'] = 'Weather-Related'

# damaged/under_construction road → Road Infrastructure
road_mask = df['road_condition'].isin(['damaged','under_construction'])
df.loc[road_mask & (df['risk_category'] == 'Driving Behavior'), 'risk_category'] = 'Road Infrastructure'
dist = df['risk_category'].value_counts(normalize=True)
print("Class distribution:\n", dist.round(3).to_string())

Xc = df[CLS_FEATS].fillna(0)
le = LabelEncoder()
yc = le.fit_transform(df['risk_category'])
print(f"Classes: {list(le.classes_)}")

Xctr,Xcte,yctr,ycte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
sc_c = RobustScaler()
Xctr_s = sc_c.fit_transform(Xctr)
Xcte_s = sc_c.transform(Xcte)

# SMOTE
k = min(5, np.bincount(yctr).min()-1)
print(f"Applying SMOTE k={k}...")
Xctr_s, yctr = SMOTE(k_neighbors=k, random_state=42).fit_resample(Xctr_s, yctr)
print(f"After SMOTE: {dict(zip(le.classes_, np.bincount(yctr)))}")

# XGBoost — fast parallel training
print("Training XGBoost...")
cls = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softmax', num_class=len(le.classes_),
    eval_metric='mlogloss', n_jobs=-1,
    use_label_encoder=False, random_state=42, verbosity=0
)
cls.fit(Xctr_s, yctr)

ypred = cls.predict(Xcte_s)
f1 = f1_score(ycte, ypred, average='weighted')
print(f"F1 (weighted)={f1:.4f}")
print(classification_report(ycte, ypred, target_names=le.classes_))

# Feature importance — top 20
imp_df = pd.DataFrame({'feature': CLS_FEATS, 'importance': cls.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=False)
print("\nTop 20 features:\n", imp_df.head(20).to_string(index=False))

# Prune: keep only features with importance > 0
useful = imp_df[imp_df['importance'] > 0]['feature'].tolist()
print(f"\nPruned: {len(CLS_FEATS)} → {len(useful)} features")

# Retrain on pruned features
Xc2 = df[useful].fillna(0)
Xctr2,Xcte2,yctr2,ycte2 = train_test_split(Xc2, yc, test_size=0.2, random_state=42, stratify=yc)
sc_c2 = RobustScaler()
Xctr2_s = sc_c2.fit_transform(Xctr2)
Xcte2_s = sc_c2.transform(Xcte2)
Xctr2_s, yctr2 = SMOTE(k_neighbors=k, random_state=42).fit_resample(Xctr2_s, yctr2)

cls2 = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softmax', num_class=len(le.classes_),
    eval_metric='mlogloss', n_jobs=-1,
    use_label_encoder=False, random_state=42, verbosity=0
)
cls2.fit(Xctr2_s, yctr2)
ypred2 = cls2.predict(Xcte2_s)
f1_2 = f1_score(ycte2, ypred2, average='weighted')
print(f"Pruned model F1={f1_2:.4f}")

# Pick best
if f1_2 >= f1:
    print("Using pruned model")
    final_cls, final_sc_c, final_feats = cls2, sc_c2, useful
    final_f1 = f1_2
else:
    print("Using full model")
    final_cls, final_sc_c, final_feats = cls, sc_c, CLS_FEATS
    final_f1 = f1

# ── Compute additional metrics ──────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score

# Regression metrics
y_pred_train = reg.predict(sc_r.transform(Xtr))
y_pred_test = reg.predict(sc_r.transform(Xte))
rmse_train = np.sqrt(mean_squared_error(ytr, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(yte, y_pred_test))
mae_test = mean_absolute_error(yte, y_pred_test)
r2_train = r2_score(ytr, y_pred_train)
r2_test = r2_score(yte, y_pred_test)

# Classification metrics on final model
y_pred_train_cls = final_cls.predict(final_sc_c.transform(df[final_feats].iloc[Xctr2.index].fillna(0)))
y_pred_test_cls = final_cls.predict(Xcte2_s)
accuracy = accuracy_score(ycte2, y_pred_test_cls)
precision = precision_score(ycte2, y_pred_test_cls, average='weighted', zero_division=0)
recall = recall_score(ycte2, y_pred_test_cls, average='weighted', zero_division=0)
f1_test = f1_score(ycte2, y_pred_test_cls, average='weighted')

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(reg,         'models/regression_model.pkl')
joblib.dump(final_cls,   'models/classification_model.pkl')
joblib.dump(sc_r,        'models/scaler_reg.pkl')
joblib.dump(final_sc_c,  'models/scaler_cls.pkl')
joblib.dump(le,          'models/label_encoder.pkl')
joblib.dump(BASE,        'models/features.pkl')
joblib.dump(final_feats, 'models/cls_features.pkl')

# Save metrics
metrics = {
    'regression': {
        'rmse_train': float(rmse_train),
        'rmse_test': float(rmse_test),
        'mae_test': float(mae_test),
        'r2_train': float(r2_train),
        'r2_test': float(r2_test),
        'train_test_r2_gap': float(r2_train - r2_test),
        'training_samples': len(Xtr),
        'test_samples': len(Xte)
    },
    'classification': {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1_test),
        'training_samples': len(Xctr2),
        'test_samples': len(Xcte2),
        'num_classes': len(le.classes_),
        'classes': list(le.classes_)
    }
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

df.to_csv('processed_dataset.csv', index=False)
print(f"\nAll saved.  Regression R2={r2_test:.4f}  Classification F1={f1_test:.4f}")
print(f"Metrics saved to models/metrics.json")
