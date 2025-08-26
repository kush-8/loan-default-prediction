import pandas as pd
import numpy as np
import json
import yaml
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer

# --- 1. Custom Feature Engineering Functions ---

# Function to aggregate bureau data

def feature_engineer_bureau_data(df, config):
    """
    Processes bureau.csv and bureau_balance.csv to create aggregated features for each SK_ID_CURR.
    """
    bureau = pd.read_csv(config['data_paths']['bureau'])
    bureau_balance = pd.read_csv(config['data_paths']['bureau_balance'])

    # --- Process bureau_balance.csv ---
    bb_cat = pd.get_dummies(bureau_balance, columns=['STATUS'], drop_first=True)
    bb_agg = bb_cat.groupby('SK_ID_BUREAU').agg(['mean', 'sum', 'var'])
    bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
    
    # --- Process bureau ---
    # Categorical aggregations
    bureau_cat_cols = bureau.select_dtypes(include='object').columns
    bureau[bureau_cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(bureau[bureau_cat_cols])
    bureau_cat_agg = pd.get_dummies(bureau.select_dtypes('object'), columns=bureau.select_dtypes('object').columns)
    bureau_cat_agg['SK_ID_CURR'] = bureau['SK_ID_CURR']
    bureau_cat_agg = bureau_cat_agg.groupby('SK_ID_CURR').agg(['mean', 'sum'])
    bureau_cat_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_cat_agg.columns.tolist()])
    
    # Numerical aggregations
    bureau_num_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['count', 'mean', 'max', 'min', 'sum'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max', 'sum'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'AMT_ANNUITY': ['mean', 'sum', 'max'],
    })
    bureau_num_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_num_agg.columns.tolist()])
    
    # Merge all new features back to the main dataframe
    df = df.join(bureau_num_agg, how='left', on='SK_ID_CURR')
    df = df.join(bureau_cat_agg, how='left', on='SK_ID_CURR')
    
    return df


# Function to aggregate previous application data

def feature_engineer_previous_data(df, config):
    """
    Processes previous_application and its related tables to create a comprehensive set of new features.
    """
    # Load all relevant tables
    prev = pd.read_csv(config['data_paths']['previous_application'])
    installments = pd.read_csv(config['data_paths']['installments_payments'])
    pos_cash = pd.read_csv(config['data_paths']['POS_CASH_balance'])
    credit_card = pd.read_csv(config['data_paths']['credit_card_balance'])

    # --- Process installments_payments ---
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    installments_agg = installments.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'PAYMENT_PERC': ['mean', 'sum', 'max', 'min'],
        'PAYMENT_DIFF': ['mean', 'sum', 'max', 'min'],
        'AMT_INSTALMENT': ['mean', 'sum', 'max'],
        'AMT_PAYMENT': ['mean', 'sum', 'max'],
    })
    installments_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()])
    df = df.join(installments_agg, how='left', on='SK_ID_CURR')

    # --- Process POS_CASH_balance ---
    pos_cash_agg = pos_cash.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['mean', 'max', 'min', 'count'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum', 'min']
    })
    pos_cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
    df = df.join(pos_cash_agg, how='left', on='SK_ID_CURR')

    # --- Process credit_card_balance ---
    credit_card_agg = credit_card.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['mean', 'sum', 'max', 'min'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'sum', 'max'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'sum', 'max'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum', 'max'],
        'SK_DPD': ['mean', 'max', 'sum']
    })
    credit_card_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in credit_card_agg.columns.tolist()])
    df = df.join(credit_card_agg, how='left', on='SK_ID_CURR')

    # --- Process previous_application ---
    prev_cat_cols = prev.select_dtypes(include='object').columns
    prev[prev_cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(prev[prev_cat_cols])
    prev_cat = pd.get_dummies(prev.select_dtypes('object'), columns=prev.select_dtypes('object').columns)
    prev_cat['SK_ID_CURR'] = prev['SK_ID_CURR']
    prev_cat_agg = prev_cat.groupby('SK_ID_CURR').agg(['mean', 'sum'])
    prev_cat_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in prev_cat_agg.columns.tolist()])
    
    prev_num_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['mean', 'sum', 'max', 'min'],
        'AMT_CREDIT': ['mean', 'sum', 'max', 'min'],
        'AMT_GOODS_PRICE': ['mean', 'sum', 'max'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_DECISION': ['mean', 'max', 'min']
    })
    prev_num_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_num_agg.columns.tolist()])
    
    df = df.join(prev_num_agg, how='left', on='SK_ID_CURR')
    df = df.join(prev_cat_agg, how='left', on='SK_ID_CURR')
    
    return df

# --- 2. Final Scikit-Learn Transformer ---

class FullFeatureEngineering(BaseEstimator, TransformerMixin):
    """Applies all fe, cleaning and feature selection steps."""
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        # --- Handle anomalies and create flags ---
        X_copy['DAYS_EMPLOYED_ANOMALY'] = (X_copy['DAYS_EMPLOYED'] == 365243)
        X_copy['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        X_copy['FLAG_OWN_CAR'] = X_copy['OWN_CAR_AGE'].notna().astype(int)

        # --- Create new ratio and time-based features ---
        X_copy['YEARS_BIRTH'] = X_copy['DAYS_BIRTH'] / -365
        X_copy['CREDIT_INCOME_PERCENT'] = X_copy['AMT_CREDIT'] / X_copy['AMT_INCOME_TOTAL']
        X_copy['ANNUITY_INCOME_PERCENT'] = X_copy['AMT_ANNUITY'] / X_copy['AMT_INCOME_TOTAL']
        X_copy['PAYMENT_RATE'] = X_copy['AMT_ANNUITY'] / X_copy['AMT_CREDIT']

        # --- Create binned features ---
        X_copy['INCOME_QUANTILE_BINS'] = pd.qcut(X_copy['AMT_INCOME_TOTAL'], q=5, labels=False, duplicates='drop')
        X_copy['CREDIT_QUANTILE_BINS'] = pd.qcut(X_copy['AMT_CREDIT'], q=5, labels=False, duplicates='drop')
        X_copy['GOODS_PRICE_QUANTILE_BINS'] = pd.qcut(X_copy['AMT_GOODS_PRICE'], q=5, labels=False, duplicates='drop')
        X_copy['ANNUITY_QUANTILE_BINS'] = pd.qcut(X_copy['AMT_ANNUITY'], q=5, labels=False, duplicates='drop')

        # Select top features of main table
        with open(self.config['model_assets']['main_table_top_features'], 'r') as f:
            main_table_top_features = json.load(f)
        
        main_table_final_cols = ['SK_ID_CURR'] + [col for col in main_table_top_features if col in X_copy.columns]
        X_copy.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_copy.columns]
        X_copy = X_copy[main_table_final_cols]

        # advanced feature engineering(historical data aggregation)
        X_copy = feature_engineer_bureau_data(X_copy, self.config)
        X_copy = feature_engineer_previous_data(X_copy, self.config)
        
        X_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

        # clean column names
        X_copy.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_copy.columns]

        # Select top features
        with open(self.config['model_assets']['final_top_features'], 'r') as f:
            top_features = json.load(f)
        
        required_cols = ['SK_ID_CURR']
        final_cols = required_cols + [col for col in top_features if col in X_copy.columns]
        
        return X_copy[final_cols]
    

# --- 3. Function to Create the Standard Preprocessing Pipeline ---    
def create_preprocessor(numerical_cols, categorical_cols):
    """Defines and returns the ColumnTransformer for standard preprocessing."""
    
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer()),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
    return preprocessor

# --- 4. Test Block ---
if __name__ == '__main__':
    # This block runs only when you execute `python src/preprocessing.py`
    
    print("--- Running preprocessing.py as a standalone script for testing ---")
    
    # Load config to get the main data path
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    app_train = pd.read_csv(config['data_paths']['application_train'])
    
    # --- Test the custom feature engineering ---
    feature_engineering_transformer = FullFeatureEngineering()
    df_engineered = feature_engineering_transformer.fit_transform(app_train)
    print(f"\nShape after custom feature engineering: {df_engineered.shape}")

    # --- Test the standard preprocessor ---
    # Identify column types from the engineered dataframe
    numerical_cols = df_engineered.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_engineered.select_dtypes(include='object').columns.tolist()
    
    # Remove ID and column from the list for preprocessing
    if 'SK_ID_CURR' in numerical_cols:
        numerical_cols.remove('SK_ID_CURR')
        
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    # Apply the preprocessor
    # We only need the features (X) for this test
    X = df_engineered.drop(columns=['SK_ID_CURR'])
    X_processed = preprocessor.fit_transform(X)
    
    print(f"\nShape after standard preprocessing (imputation, scaling, encoding): {X_processed.shape}")
    print("\nPreprocessing script test completed successfully!")