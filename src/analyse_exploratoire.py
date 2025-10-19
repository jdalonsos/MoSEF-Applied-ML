import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


def display_missing_values(df):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    print("\n Les taux de valeurs manquantes (en %) : \n", missing)


def impute_by_regression(df):
    try:
        numeric_cols = col_numericals
    except NameError as e:
        raise NameError(
            "The global variable `col_numericals` must be defined to use impute_by_regression."
        ) from e
    df_num = df[numeric_cols]
    imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns, index=df.index)
    for n in numeric_cols:
        if n in ("age", "exp"):
            df[f"{n}_imputation_regression"] = df[n].fillna(df_imputed[n].round()).astype("Int64")
        else:
            df[f"{n}_imputation_regression"] = df[n].fillna(df_imputed[n])


def is_outlier(df, column):
    Q1 = np.quantile(df[column], 0.25)
    Q3 = np.quantile(df[column], 0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    print(f"pour la variable {column} , limite inf = {limite_inf}")
    print(f"pour la variable {column} , limite sup = {limite_sup}")
    return ((df[column] > limite_sup) | (df[column] < limite_inf)).astype(int)


def display_missing_values_bis(df):
    missing_prct = df.isna().mean() * 100
    missing_nb = df.isna().sum()
    return (
        pd.DataFrame({"missing_nb": missing_nb, "missing_prct": missing_prct})
        .sort_values("missing_prct", ascending=False)
    )