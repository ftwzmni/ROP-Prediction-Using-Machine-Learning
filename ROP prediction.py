import warnings
warnings.filterwarnings('ignore')

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
from scipy.stats import chi2
import random
from scipy.optimize import differential_evolution
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Optional: SHAP for model explainability (will be used if installed)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 0) Paper Models: TPR / PRB / PRF
class _Poly2Reg:
    def __init__(self):
        self.coef_ = None
        self.n_features_ = None

    def _design(self, X):
        # X: (n, d)
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        cols = [np.ones((n, 1))]   # bias term a0
        cols.append(X)             # linear
        cols.append(X ** 2)        # squares

        inter = []
        for i in range(d):
            for j in range(i + 1, d):
                inter.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if len(inter) > 0:
            cols.append(np.hstack(inter))

        return np.hstack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.n_features_ = X.shape[1]
        Phi = self._design(X)
        # least squares
        self.coef_, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Phi = self._design(X)
        yhat = Phi @ self.coef_
        return yhat.ravel()


class _TPRNode:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "model")
    def __init__(self):
        self.is_leaf = True
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.model = None


class TPRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 min_leaf=10,
                 max_depth=10,
                 random_state=42,
                 strategy="optimized",
                 k_features=None):
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.strategy = strategy
        self.k_features = k_features

        self.root_ = None

    def _split_points(self, col_min, col_max):
        # 10 points: L + t*(M-L)/11 for t=1..10 (paper)
        if not np.isfinite(col_min) or not np.isfinite(col_max) or col_max <= col_min:
            return []
        step = (col_max - col_min) / 11.0
        return [col_min + t * step for t in range(1, 11)]

    @staticmethod
    def _mse(y, yhat):
        y = np.asarray(y, dtype=float).ravel()
        yhat = np.asarray(yhat, dtype=float).ravel()
        return float(np.mean((y - yhat) ** 2))

    def _node_score(self, X_train, y_train, X_test, y_test):
        mdl = _Poly2Reg().fit(X_train, y_train)
        yhat = mdl.predict(X_test)
        return self._mse(y_test, yhat), mdl

    def _best_split(self, X, y, rng):
        n, d = X.shape
        if n < 2 * self.min_leaf:
            return None

        # Split node data into Strain (80%) and Stest (20%) similar to paper
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, n // 5)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        base_mse, _ = self._node_score(Xtr, ytr, Xte, yte)

        feats = np.arange(d)
        if self.strategy == "randomized":
            if self.k_features is None:
                # default as in RF: log2(d) rounded up
                k = int(np.ceil(np.log2(max(d, 2))))
            else:
                k = int(self.k_features)
            k = max(1, min(k, d))
            candidate = rng.choice(feats, size=k, replace=False)
            # paper-like: choose ONE variable from k
            feats = np.array([rng.choice(candidate)])

        best = (None, None, np.inf)  # (feature, threshold, split_mse)

        for j in feats:
            col = Xtr[:, j]
            pts = self._split_points(np.min(col), np.max(col))
            if len(pts) == 0:
                continue

            for th in pts:
                left_tr = Xtr[:, j] < th
                right_tr = ~left_tr

                if left_tr.sum() < self.min_leaf or right_tr.sum() < self.min_leaf:
                    continue

                left_te = Xte[:, j] < th
                right_te = ~left_te
                if left_te.sum() == 0 or right_te.sum() == 0:
                    continue

                mse_left, _ = self._node_score(Xtr[left_tr], ytr[left_tr], Xte[left_te], yte[left_te])
                mse_right, _ = self._node_score(Xtr[right_tr], ytr[right_tr], Xte[right_te], yte[right_te])

                split_mse = (left_te.sum() * mse_left + right_te.sum() * mse_right) / (left_te.sum() + right_te.sum())

                if split_mse < best[2]:
                    best = (int(j), float(th), float(split_mse))

        if best[0] is None or best[2] >= base_mse:
            return None

        return best

    def _build(self, X, y, depth, rng):
        node = _TPRNode()
        node.model = _Poly2Reg().fit(X, y)

        if depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            return node

        best = self._best_split(X, y, rng)
        if best is None:
            return node

        feat, th, _ = best
        mask = X[:, feat] < th
        if mask.sum() < self.min_leaf or (~mask).sum() < self.min_leaf:
            return node

        node.is_leaf = False
        node.feature = feat
        node.threshold = th
        node.left = self._build(X[mask], y[mask], depth + 1, rng)
        node.right = self._build(X[~mask], y[~mask], depth + 1, rng)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        rng = np.random.RandomState(self.random_state)
        self.root_ = self._build(X, y, depth=0, rng=rng)
        return self

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.model.predict(x.reshape(1, -1))[0]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])


class PRBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_estimators=25,
                 alpha_frac=0.95,
                 min_leaf=10,
                 max_depth=10,
                 random_state=42):
        self.n_estimators = n_estimators
        self.alpha_frac = alpha_frac
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self.models_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n = len(y)
        alpha = int(np.round(self.alpha_frac * n))
        alpha = max(2 * self.min_leaf, min(alpha, n))

        rng = np.random.RandomState(self.random_state)
        self.models_ = []

        for _ in range(int(self.n_estimators)):
            idx = rng.choice(np.arange(n), size=alpha, replace=True)
            mdl = TPRRegressor(
                min_leaf=self.min_leaf,
                max_depth=self.max_depth,
                random_state=int(rng.randint(0, 10**9)),
                strategy="optimized",
                k_features=None
            )
            mdl.fit(X[idx], y[idx])
            self.models_.append(mdl)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return preds.mean(axis=1)


class PRFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_estimators=25,
                 alpha_frac=0.95,
                 k_features=2,
                 min_leaf=10,
                 max_depth=10,
                 random_state=42):
        self.n_estimators = n_estimators
        self.alpha_frac = alpha_frac
        self.k_features = k_features
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self.models_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n = len(y)
        alpha = int(np.round(self.alpha_frac * n))
        alpha = max(2 * self.min_leaf, min(alpha, n))

        rng = np.random.RandomState(self.random_state)
        self.models_ = []

        for _ in range(int(self.n_estimators)):
            idx = rng.choice(np.arange(n), size=alpha, replace=True)
            mdl = TPRRegressor(
                min_leaf=self.min_leaf,
                max_depth=self.max_depth,
                random_state=int(rng.randint(0, 10**9)),
                strategy="randomized",
                k_features=int(self.k_features)
            )
            mdl.fit(X[idx], y[idx])
            self.models_.append(mdl)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return preds.mean(axis=1)


# 2. Load dataset
FILE = "data.xlsx"
SHEET = 0

try:
    df = pd.read_excel(FILE, sheet_name=SHEET)
except Exception as e:
    raise SystemExit(f"Failed to read {FILE}: {e}")

# drop No. column if exists (case-insensitive)
for col in df.columns:
    if str(col).strip().lower() in ["no.", "no", "number", "index"]:
        df.drop(columns=[col], inplace=True)
        break

print(f"Dataset shape: {df.shape}")

# 3. Quick data check and cleaning
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:\n", df.dtypes)

missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0])

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if df[num_cols].isnull().any().any():
    print("\nImputing numeric missing values with median")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

if len(cat_cols) > 0 and df[cat_cols].isnull().any().any():
    print("\nImputing categorical missing values with mode")
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])


# 4. Descriptive statistics (stronger)
pd.set_option('display.max_columns', None)

desc = df.describe().T
desc['skew'] = df[num_cols].skew()
desc['kurtosis'] = df[num_cols].kurtosis()

print("\nEnhanced descriptive statistics:")
print(desc)


# 5. Target check and possible transform
TARGET = 'ROP'
if TARGET not in df.columns:
    raise SystemExit(f"Target column '{TARGET}' not found in dataset")

print("\nTarget distribution summary:")
print(df[TARGET].describe())

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], kde=True)
plt.title('ROP distribution')
plt.tight_layout()
plt.savefig('ROP_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

target_skew = df[TARGET].skew()
print(f"Target skewness: {target_skew:.3f}")

# (kept as you wrote: not auto-transforming, only optional if you enable later)
if (df[TARGET] > 0).all():
    pt = PowerTransformer(method='yeo-johnson')
else:
    pt = PowerTransformer(method='yeo-johnson')


# 6. EDA visualizations
EDA_SAMPLE = df.sample(n=min(1000, len(df)), random_state=RANDOM_STATE)

numeric = num_cols.copy()
if TARGET in numeric:
    numeric.remove(TARGET)

corr = df[numeric + [TARGET]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix')
plt.tight_layout()
plt.savefig('Correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

MAX_PAIR = 6
pair_features = numeric[:MAX_PAIR] + [TARGET]
if len(pair_features) > 1:
    g = sns.pairplot(
        EDA_SAMPLE[pair_features],
        diag_kind='kde',
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.6}, 'line_kws': {'color': 'red', 'linewidth': 2}}
    )
    plt.suptitle('Pairplot (sample)', y=1.02)
    g.fig.tight_layout()
    g.fig.savefig('Pairplot (sample).png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

for col in numeric:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')
    plt.tight_layout()
    plt.savefig(f'Boxplot_{col}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

for col in numeric:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df[TARGET])
    sns.regplot(x=df[col], y=df[TARGET], scatter=False, lowess=True)
    plt.xlabel(col)
    plt.ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(f"{col}_vs_{TARGET}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# 7. Outlier detection (IQR + Mahalanobis)
outlier_info = {}
for col in numeric:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_info[col] = int(n_out)

print('\nOutlier counts (IQR method):')
for kk, vv in outlier_info.items():
    print(f"{kk}: {vv}")

df['n_outlier_flags'] = 0
for col in numeric:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df['n_outlier_flags'] += ((df[col] < lower) | (df[col] > upper)).astype(int)

print('\nRows with >=3 outlier flags (IQR-based):', int((df['n_outlier_flags'] >= 3).sum()))

X_m = df[numeric].dropna()
X_vals = X_m.values

mu = X_vals.mean(axis=0)
cov = np.cov(X_vals, rowvar=False)

try:
    inv_cov = np.linalg.inv(cov)
except np.linalg.LinAlgError:
    inv_cov = np.linalg.pinv(cov)

m_dist2 = np.einsum('ij,jk,ik->i', (X_vals - mu), inv_cov, (X_vals - mu))

mahal_k = X_vals.shape[1]
threshold = chi2.ppf(0.99, df=mahal_k)

df['mahal_dist2'] = np.nan
df.loc[X_m.index, 'mahal_dist2'] = m_dist2
df['mahal_outlier'] = df['mahal_dist2'] > threshold

print('\nMahalanobis outliers (alpha=0.01):', int(df['mahal_outlier'].sum()))

outlier_rows = df[df['mahal_outlier']].copy()
print("\nMahalanobis Outlier Rows (index + distance):")
print(outlier_rows[['mahal_dist2']])

print("\nDifference of each outlier from non-outlier mean (signed difference):")
feature_means = X_m.mean()
outlier_diff = {}

for idx, row in outlier_rows[numeric].iterrows():
    diffs = row - feature_means
    outlier_diff[idx] = diffs
    print(f"\nOutlier index {idx}:")
    print(diffs)

print("\nFeature with highest contribution for each outlier:")
for idx, diffs in outlier_diff.items():
    abs_diffs = diffs.abs()
    key_feature = abs_diffs.idxmax()
    signed_value = diffs[key_feature]
    sign_str = "+" if signed_value > 0 else "-"
    print(f"Outlier {idx} → Most influential feature: {key_feature} (difference = {sign_str}{abs(signed_value):.4f})")

df['outlier_score'] = df['n_outlier_flags'].fillna(0) + df['mahal_outlier'].fillna(False).astype(int)

n_before = len(df)
df = df[df['mahal_outlier'] == False].copy()
n_after = len(df)

print(f"\nRemoved {n_before - n_after} Mahalanobis outliers.")
print(f"New dataset shape: {df.shape}")


# 8. Feature selection / engineering
preferred_features = ["Q(m)  GPM", "p(g)", "T(g)", "RPM", "D(b)", "W.O.B", "H"]
features = [f for f in preferred_features if f in df.columns]
if len(features) == 0:
    features = numeric.copy()

print('\nUsing features:', features)

X = df[features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# 9. Modeling: pipelines + cross-validation + hyperparam tuning
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    # ('pt', PowerTransformer(method='yeo-johnson'))  # optional
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, features)
])

models_and_params = {
    'LinearRegression': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())]),
        'params': {}
    },
    'Ridge': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', Ridge())]),
        'params': {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Lasso': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', Lasso(max_iter=5000))]),
        'params': {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    },
    'RandomForest': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 5, 10]
        }
    },
    'GradientBoosting': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [2, 3, 4]
        }
    },
    'AdaBoost': {
        'pipeline': Pipeline(steps=[('pre', preprocessor), ('model', AdaBoostRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1.0]
        }
    },
'NN_Relu': {
    'pipeline': Pipeline(steps=[('pre', preprocessor),
                                ('model', MLPRegressor(
                                    max_iter=5000,
                                    random_state=RANDOM_STATE,
                                    early_stopping=True,
                                    n_iter_no_change=30
                                ))]),
    'params': {
        'model__hidden_layer_sizes': [
            (4,), (8,), (12,), (16,), (24,),
            (8,4), (12,6), (16,8), (20,10), (24,12),
            (16,8,4), (12,8,4), (24,12,6), 
            (64, 32), (32, 16), (15, 7), (64, 32, 32)
            ],
        'model__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'model__learning_rate_init': [0.001, 0.01]
        }
},
'NN_Sigmoid': {
    'pipeline': Pipeline(steps=[('pre', preprocessor),
                                ('model', MLPRegressor(
                                    activation='logistic',
                                    solver='adam',
                                    max_iter=8000,
                                    random_state=RANDOM_STATE,
                                    early_stopping=True,
                                    validation_fraction=0.2,
                                    n_iter_no_change=40,
                                    tol=1e-5
                                ))]),
    'params': {
        'model__hidden_layer_sizes': [
            (4,), (8,), (12,), (16,),
            (8, 4), (12, 6), (16, 8),
            (16, 8, 4)
        ],

        'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'model__learning_rate_init': [1e-4, 5e-4, 1e-3, 5e-3],
        'model__batch_size': [16, 32]
    }
},

    'TPR': {
        'pipeline': Pipeline(steps=[('pre', preprocessor),
                                    ('model', TPRRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__max_depth': [6, 10],
            'model__min_leaf': [5, 10],
            'model__strategy': ['optimized']
        }
    },
    'PRB': {
        'pipeline': Pipeline(steps=[('pre', preprocessor),
                                    ('model', PRBRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__n_estimators': [25, 50],
            'model__alpha_frac': [0.90, 0.95, 1.00],
            'model__max_depth': [6, 10],
            'model__min_leaf': [5, 10]
        }
    },
    'PRF': {
        'pipeline': Pipeline(steps=[('pre', preprocessor),
                                    ('model', PRFRegressor(random_state=RANDOM_STATE))]),
        'params': {
            'model__n_estimators': [25, 50],
            'model__alpha_frac': [0.90, 0.95, 1.00],
            'model__k_features': [1, 2, 3],
            'model__max_depth': [6, 10],
            'model__min_leaf': [5, 10]
        }
    }
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, spec in models_and_params.items():
    print(f"\nTraining and tuning: {name}")
    pipeline = spec['pipeline']
    params = spec['params']

    if params:
        gs = GridSearchCV(pipeline, params, cv=cv, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        print(f"Best params for {name}: {gs.best_params_}")
        model_to_eval = best
    else:
        pipeline.fit(X_train, y_train)
        model_to_eval = pipeline

    y_pred = model_to_eval.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {'r2': r2, 'mse': mse, 'mae': mae, 'model': model_to_eval}
    print(f"{name} --> R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")


# 10. Select best model and save
best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest model by R2: {best_model_name}")

joblib.dump(best_model, 'best_model.joblib')
print('Saved best model to best_model.joblib')


# 11. Residual analysis
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual ROP")
plt.ylabel("Predicted ROP")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=300)
plt.show()
plt.close()
print("Saved predicted vs actual plot to predicted_vs_actual.png")

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted ROP")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.savefig("residuals_vs_predicted.png", dpi=300)
plt.show()
plt.close()
print("Saved residuals vs predicted plot to residuals_vs_predicted.png")

plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='purple', bins=20)
plt.xlabel("Residuals")
plt.title("Residuals Distribution")
plt.tight_layout()
plt.savefig("residuals_distribution.png", dpi=300)
plt.show()
plt.close()
print("Saved residuals distribution plot to residuals_distribution.png")

# 12) Overfitting Check (Learning Curve + Train vs Test R² Plot)
def _resolve_estimator(model):
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def _ensure_fitted(model, X_train, y_train):
    model = _resolve_estimator(model)
    try:
        model.predict(X_train.iloc[:2] if hasattr(X_train, "iloc") else X_train[:2])
    except Exception:
        print("Model not fitted yet → fitting now...")
        model.fit(X_train, y_train)
    return model


def plot_learning_curve_r2(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
                           random_state=42, save_path="learning_curve_r2.png"):
    model = _resolve_estimator(model)


    if isinstance(cv, int):
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_obj = cv

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv_obj,
        scoring="r2",
        train_sizes=train_sizes,
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std  = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, train_mean, "o-", label="Train R²")
    plt.plot(train_sizes_abs, val_mean,  "o-", label="CV (Validation) R²")

    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes_abs, val_mean - val_std,     val_mean + val_std,     alpha=0.15)

    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title("Learning Curve (Overfitting Check)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved Learning Curve as: {save_path}")

    return {
        "train_sizes": train_sizes_abs,
        "train_mean": train_mean,
        "train_std": train_std,
        "val_mean": val_mean,
        "val_std": val_std
    }


def train_test_r2_with_plot(model, X_train, X_test, y_train, y_test,
                            gap_threshold=0.15,
                            save_path="train_vs_test_r2.png"):
    model = _ensure_fitted(model, X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test, y_test_pred)
    gap = r2_train - r2_test

    print("Train vs Test R² Comparison")
    print(f"Train R² : {r2_train:.4f}")
    print(f"Test  R² : {r2_test:.4f}")
    print(f"Gap (Train - Test) = {gap:.4f}")

    if gap > gap_threshold:
        print("Model is likely OVERFITTING!")
    else:
        print("Model generalization looks OK.")


    plt.figure(figsize=(6, 5))
    labels = ["Train R²", "Test R²"]
    values = [r2_train, r2_test]

    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("R² Score")
    plt.title("Train vs Test R² (Overfitting Check)")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved Train vs Test R² plot as: {save_path}")

    return r2_train, r2_test, 

plot_learning_curve_r2(best_model, X, y, cv=5)
train_test_r2_with_plot(best_model, X_train, X_test, y_train, y_test)

# 13. Feature importance / coefficients (safe + generalized)
print('\nFeature importances / coefficients (best model):')

# try to find final estimator for pipelines
try:
    model_core = best_model.named_steps['model']
except Exception:
    model_core = best_model

try:
    if hasattr(model_core, 'feature_importances_'):
        importances = model_core.feature_importances_
        for f, imp in zip(features, importances):
            print(f"{f}: {imp:.4f}")

    elif hasattr(model_core, 'coef_'):
        coefs = model_core.coef_
        # in case of multi-output: flatten
        coefs = np.asarray(coefs).ravel()
        for f, c in zip(features, coefs):
            print(f"{f}: {c:.4f}")

    else:
        print("Model does not provide simple importances/coefficients (OK for paper ensembles).")

except Exception as e:
    print('Could not extract importances/coefficients:', e)

# 14. SHAP explanation (robust)
print("\nRunning SHAP analysis...")

# For custom ensembles (TPR/PRB/PRF), SHAP may not work directly. We'll skip safely.
CUSTOM_PAPER_MODELS = (TPRRegressor, PRBRegressor, PRFRegressor)

if SHAP_AVAILABLE:
    try:
        # get model for shap
        model_for_shap = model_core

        if isinstance(model_for_shap, CUSTOM_PAPER_MODELS):
            print("SHAP skipped: custom paper-based model (TPR/PRB/PRF) not directly supported by SHAP.")
        else:
            X_train_df = pd.DataFrame(X_train, columns=features)
            X_test_df = pd.DataFrame(X_test, columns=features)

            # Tree models
            if hasattr(model_for_shap, "feature_importances_"):
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values = explainer.shap_values(X_test_df)
                shap.summary_plot(shap_values, X_test_df, show=False)
            else:
                # KernelExplainer fallback
                background = shap.kmeans(X_train_df, 10)
                explainer = shap.KernelExplainer(model_for_shap.predict, background)
                shap_values = explainer.shap_values(X_test_df.iloc[:100])
                shap.summary_plot(shap_values, X_test_df.iloc[:100], show=False)

            plt.tight_layout()
            plt.savefig("shap_summary.png", dpi=300)
            plt.show()
            plt.close()
            print("Saved SHAP summary plot to shap_summary.png")

    except Exception as e:
        print("SHAP analysis failed:", e)

else:
    print('SHAP not available. To enable: pip install shap')


# 15. Export results summary as CSV, Image and Bar plots
summary_rows = []
for name, res in results.items():
    summary_rows.append({'model': name, 'r2': res['r2'], 'mse': res['mse'], 'mae': res['mae']})

summary_df = pd.DataFrame(summary_rows).sort_values('r2', ascending=False)
print('\nModel comparison:')
print(summary_df)

summary_df.to_csv('model_comparison.csv', index=False)
print('Saved model comparison to model_comparison.csv')

fig, ax = plt.subplots(figsize=(6, 2 + len(summary_df) * 0.4))
ax.axis('off')
table_data = [[row['model'], f"{row['r2']:.3f}", f"{row['mse']:.3f}"] for _, row in summary_df.iterrows()]
col_labels = ["Model", "R²", "MSE"]
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Model Performance Summary", pad=20)
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print('Saved model comparison image to model_comparison.png')

plt.figure(figsize=(8, 5))
plt.bar(summary_df['model'], summary_df['r2'])
plt.ylabel('R²')
plt.ylim(0, 1)
plt.title('R² of Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("model_r2.png", dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary_df['model'], summary_df['mse'])
plt.ylabel('MSE')
plt.title('MSE of Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("model_mse.png", dpi=300)
plt.show()
plt.close()

models = summary_df['model']
r2_values = summary_df['r2']
mse_values = summary_df['mse']

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width/2, r2_values, width, label='R²')
plt.bar(x + width/2, mse_values, width, color='pink', label='MSE')
plt.ylabel('Value')
plt.title('R² and MSE Comparison of Models')
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("model_r2_mse_combined.png", dpi=300)
plt.show()
plt.close()

# -------------------------------
# 16. Optimization: best operating point (Operations Research style)
# -------------------------------
print("\n" + "=" * 70)
print("Global optimization on best model (Operations Research style)")
print("=" * 70)

feature_mins = X.min()
feature_maxs = X.max()

print("\nFeature ranges used as bounds in optimization:")
for f in features:
    print(f"  {f}: [{feature_mins[f]:.4f}, {feature_maxs[f]:.4f}]")

def objective(xvec):
    x_df = pd.DataFrame([xvec], columns=features)
    y_hat = best_model.predict(x_df)[0]
    return -y_hat  # maximize ROP

bounds = [(feature_mins[f], feature_maxs[f]) for f in features]

result = differential_evolution(
    objective,
    bounds,
    maxiter=500,
    tol=1e-6,
    seed=RANDOM_STATE
)

best_x = result.x
best_rop = -result.fun

print("\nOptimization finished.")
print(f"Success flag       : {result.success}")
print(f"Optimizer message  : {result.message}")

print("\nOptimal operating point (continuous search inside [min, max] of each feature):")
for f, val in zip(features, best_x):
    print(f"  {f:10s} = {val:10.4f}   (range [{feature_mins[f]:.4f}, {feature_maxs[f]:.4f}])")

print(f"\nPredicted MAX ROP by best_model ({best_model_name}): {best_rop:.4f}")

print("\n" + "-" * 70)
print("Comparing with the best existing sample in the dataset (according to model):")

X_with_pred = X.copy()
X_with_pred['pred_ROP'] = best_model.predict(X)

idx_max_sample = X_with_pred['pred_ROP'].idxmax()
best_sample = X_with_pred.loc[idx_max_sample]

print(f"\nBest existing sample index: {idx_max_sample}")
print("Feature values of best existing sample:")
for f in features:
    print(f"  {f:10s} = {best_sample[f]:10.4f}")

print(f"\nPredicted ROP for best existing sample: {best_sample['pred_ROP']:.4f}")

print("\nScript finished. Check generated files including shap_summary.png (if created)")

