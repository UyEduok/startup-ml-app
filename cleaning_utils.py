import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_invalid_numbers(df):
    results = {}
    cols_to_check = ['age_first_funding_year', 'age_last_funding_year']

    for col in cols_to_check:
        if col in df.columns:
            results[col] = {
                "summary_stats": df[col].describe(),
                "missing_values": int(df[col].isnull().sum()),
                "negative_count": int((df[col] < 0).sum()),
                "negative_rows": df[df[col] < 0][[col]].head(20),
                "smallest_10": pd.DataFrame(
                    df[col].sort_values().head(10).values,
                    columns=[col]
                ),
                "largest_10": pd.DataFrame(
                    df[col].sort_values(ascending=False).head(10).values,
                    columns=[col]
                )
            }

    return results


def clean_dataset(df):
    df_clean = df.copy()

    initial_shape = df_clean.shape

    # 1. Drop irrelevant / identifier columns
    cols_to_drop = [
        'Unnamed: 0',
        'Unnamed: 6',
        'id',
        'object_id',
        'name',
        'zip_code',
        'state_code.1',
        'closed_at'
    ]

    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    after_drop_shape = df_clean.shape

    # 2. Handle date columns
    date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']

    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    if all(col in df_clean.columns for col in date_cols):
        df_clean['startup_age_days'] = (
            df_clean['first_funding_at'] - df_clean['founded_at']
        ).dt.days

        df_clean['funding_gap_days'] = (
            df_clean['last_funding_at'] - df_clean['first_funding_at']
        ).dt.days

    df_clean.drop(
        columns=[col for col in date_cols if col in df_clean.columns],
        inplace=True
    )

    # 3. Handle missing values
    fill_log = []

    milestone_cols = ['age_first_milestone_year', 'age_last_milestone_year']
    for col in milestone_cols:
        if col in df_clean.columns:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            fill_log.append(f"{col} missing values filled with median: {median_value}")

    derived_date_cols = ['startup_age_days', 'funding_gap_days']
    for col in derived_date_cols:
        if col in df_clean.columns:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            fill_log.append(f"{col} missing values filled with median: {median_value}")

    # 4. Fix invalid negative values
    invalid_cols = ['age_first_funding_year', 'age_last_funding_year']
    invalid_fix_log = []

    for col in invalid_cols:
        if col in df_clean.columns:
            median_valid = df_clean.loc[df_clean[col] >= 0, col].median()
            negative_count = int((df_clean[col] < 0).sum())
            df_clean.loc[df_clean[col] < 0, col] = median_valid
            invalid_fix_log.append(
                f"{col}: {negative_count} negative values replaced with median {median_valid}"
            )

    final_shape = df_clean.shape

    remaining_missing = df_clean.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    remaining_negative = {}
    for col in invalid_cols:
        if col in df_clean.columns:
            remaining_negative[col] = int((df_clean[col] < 0).sum())

    return {
        "df_clean": df_clean,
        "initial_shape": initial_shape,
        "after_drop_shape": after_drop_shape,
        "final_shape": final_shape,
        "fill_log": fill_log,
        "invalid_fix_log": invalid_fix_log,
        "remaining_missing": remaining_missing,
        "remaining_negative": remaining_negative,
        "columns_after_cleaning": df_clean.columns.tolist()
    }


def transform_features(df_clean):
    # Make a working copy from cleaned data
    df_model = df_clean.copy()

    initial_shape = df_model.shape

    # 1. Encode target variable
    df_model['status'] = df_model['status'].map({'acquired': 1, 'closed': 0})
    target_encoding_counts = df_model['status'].value_counts()

    # 2. Handle categorical variables
    df_model.drop(columns=['city'], inplace=True, errors='ignore')
    df_model.drop(columns=['labels'], inplace=True, errors='ignore')

    categorical_cols = ['state_code', 'category_code']
    existing_categorical_cols = [col for col in categorical_cols if col in df_model.columns]

    df_model = pd.get_dummies(df_model, columns=existing_categorical_cols, drop_first=True)

    shape_after_encoding = df_model.shape

    # 3. Define X and y
    X = df_model.drop('status', axis=1)
    y = df_model['status']

    X_shape = X.shape
    y_shape = y.shape

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    split_shapes = {
        "X_train": X_train.shape,
        "X_test": X_test.shape,
        "y_train": y_train.shape,
        "y_test": y_test.shape
    }

    # 5. Feature scaling
    numeric_cols = X_train.select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']
    ).columns

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    scaled_column_count = len(numeric_cols)

    # 6. Output checks
    train_sample = X_train.head()
    test_sample = X_test.head()

    y_train_dist = y_train.value_counts(normalize=True) * 100
    y_test_dist = y_test.value_counts(normalize=True) * 100

    return {
        "df_model": df_model,
        "initial_shape": initial_shape,
        "target_encoding_counts": target_encoding_counts,
        "shape_after_encoding": shape_after_encoding,
        "X_shape": X_shape,
        "y_shape": y_shape,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_shapes": split_shapes,
        "numeric_cols": list(numeric_cols),
        "scaled_column_count": scaled_column_count,
        "train_sample": train_sample,
        "test_sample": test_sample,
        "y_train_dist": y_train_dist,
        "y_test_dist": y_test_dist,
        "scaler": scaler
    }


def transform_features_for_prediction(df_clean):
    """
    Transform cleaned raw input data into model-ready features for prediction.
    This mirrors the training transformation logic, but:
    - does NOT require status
    - does NOT split train/test
    - does NOT scale
    """
    df_model = df_clean.copy()

    # Drop columns exactly like training transformation
    df_model.drop(columns=['city'], inplace=True, errors='ignore')
    df_model.drop(columns=['labels'], inplace=True, errors='ignore')

    # If uploaded data contains status, remove it for prediction
    df_model.drop(columns=['status'], inplace=True, errors='ignore')

    categorical_cols = ['state_code', 'category_code']
    existing_categorical_cols = [col for col in categorical_cols if col in df_model.columns]

    df_model = pd.get_dummies(df_model, columns=existing_categorical_cols, drop_first=True)

    # Convert bool columns to int if present
    bool_cols = df_model.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df_model[bool_cols] = df_model[bool_cols].astype(int)

    # Force all columns to numeric where possible
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    return df_model