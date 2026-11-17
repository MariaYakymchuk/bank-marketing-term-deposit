"""Data loading, cleaning, feature engineering and the sklearn preprocessor."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = 'y'

# Features used by the models (after feature engineering, without `duration`)
NUMERIC_FEATURES = ['age', 'campaign', 'pdays', 'previous', 'contacted_before',
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome', 'season']

# 999 in `pdays` means "the client was never contacted before"
PDAYS_NOT_CONTACTED = 999

# Similar or very rare categories are joined into one group
CATEGORY_GROUPS = {
    'education': {'basic.4y': 'basic', 'basic.6y': 'basic', 'basic.9y': 'basic', 'illiterate': 'basic'},
    'default': {'yes': 'unknown'},  # only 3 clients have 'yes'
}

MONTH_TO_SEASON = {
    'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
    'mar': 'spring', 'apr': 'spring', 'may': 'spring',
    'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
    'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn',
}


def load_data(path: str) -> pd.DataFrame:
    """Read the raw csv file (columns are separated by ';')."""
    return pd.read_csv(path, sep=';')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and turn the target into 0 / 1."""
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] == 'yes').astype(int)
    return df


def add_features(df: pd.DataFrame, keep_duration: bool = False) -> pd.DataFrame:
    """Create new features and group categories.

    `duration` is removed by default because it is only known after the call (data leakage).
    """
    data = df.copy()
    if not keep_duration:
        data = data.drop(columns=['duration'])

    data['contacted_before'] = (data['pdays'] != PDAYS_NOT_CONTACTED).astype(int)
    data['pdays'] = data['pdays'].replace(PDAYS_NOT_CONTACTED, -1)

    for col, mapping in CATEGORY_GROUPS.items():
        data[col] = data[col].replace(mapping)

    data['season'] = data['month'].map(MONTH_TO_SEASON)
    return data


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    """Stratified train / validation split. Returns X_train, X_val, y_train, y_val."""
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def clip_outliers(X_train: pd.DataFrame, X_val: pd.DataFrame, col: str, quantile: float = 0.99) -> float:
    """Limit very big values of `col`. The limit is learned on train only (no leakage).

    Changes both tables in place and returns the limit.
    """
    cap = X_train[col].quantile(quantile)
    X_train[col] = X_train[col].clip(upper=cap)
    X_val[col] = X_val[col].clip(upper=cap)
    return cap


def build_preprocessor(numeric_features=NUMERIC_FEATURES,
                       categorical_features=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical ones."""
    return ColumnTransformer(
        [
            ('num', StandardScaler(), list(numeric_features)),
            ('cat', OneHotEncoder(handle_unknown='ignore'), list(categorical_features)),
        ],
        sparse_threshold=0,  # always return a normal (dense) array - needed for SHAP
        verbose_feature_names_out=False,  # feature names without 'num__' / 'cat__'
    )
