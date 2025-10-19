import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression



def display_missing_values(df):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\n Taux de valeurs manquantes (%):\n", missing)


def impute_by_regression(df, columns_to_impute):

    
    # Keep only selected columns
    df_selected = df[columns_to_impute]
    
    # Imputer setup
    imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_selected),
        columns=df_selected.columns,
        index=df.index
    )
    
    # Add new columns with suffix only for the ones imputed
    for col in columns_to_impute:
        df[f"{col}_imputation_regression"] = df_imputed[col]
    
    return df

def impute_categorical_random_newcols(df, cols):
    np.random.seed(0)  # pour un résultat reproductible
    for col in cols:
        # créer une nouvelle colonne
        new_col = f"{col}_imputation_random"
        df[new_col] = df[col].copy()

        # repérer les valeurs manquantes
        missing_idx = df[new_col].isna()

        # imputer par tirage aléatoire parmi les valeurs existantes
        df.loc[missing_idx, new_col] = np.random.choice(df[new_col].dropna(), size=missing_idx.sum(), replace=True)
    
    return df

def is_outlier(df,column) :
    
    # 1er Quartile 
    Q1 = df[column].quantile(0.25)
    
    # 3ème Quartile 
    Q3 = df[column].quantile(0.75)
    
    # Inter-Quartile Range (IQR)
    IQR = Q3 - Q1
    
    # limites, basse & haute
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    
    # Remplace les données inférieur et supérieur à la limite par 1 et les autres par 0
    series =  ((df[column] < limite_inf) | (df[column] > limite_sup)).astype(float)
    
    return series


def display_missing_values_Sec(df):
    missing = (
        df.isnull().sum()
        .to_frame("nb_valeurs_manquantes")
        .assign(taux=lambda x: x["nb_valeurs_manquantes"] / len(df) * 100)
        .query("nb_valeurs_manquantes > 0")
        .sort_values("nb_valeurs_manquantes", ascending=False)
        .round(2)
    )
    return missing


