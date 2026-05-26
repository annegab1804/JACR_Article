
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

def fit_logistic_regression(df: pd.DataFrame, outcome_col: str,
                             predictor_cols: list) -> object:
    """Binary logistic regression via statsmodels.

    Used for:
        * Any demographic reporting (binary: 0/1)
        * PCCP authorization (binary: 0/1)
        * Multi-site validation (binary: 0/1)

    Common predictors include Decision_year (numeric), Pathway (dummies),
    Modality (dummies), Body_Region (dummies), and HQ_Region (dummies).

    Args:
        df: Pandas DataFrame containing the outcome and predictor columns.
        outcome_col: Name of the binary (0/1) target column.
        predictor_cols: List of column names to use as features. Categorical 
            columns are automatically treated as dummy variables via the formula interface.

    Returns:
        statsmodels.discrete.discrete_model.BinaryResultsWrapper: The fitted 
            logistic regression model. Call `.summary()` on the returned object to print.
    """
    import statsmodels.formula.api as smf

    # Build formula: outcome ~ C(cat1) + C(cat2) + numeric_col
    numeric_cols = [c for c in predictor_cols
                    if pd.api.types.is_numeric_dtype(df[c])]
    categ_cols   = [c for c in predictor_cols
                    if not pd.api.types.is_numeric_dtype(df[c])]

    parts = [f'C({c})' for c in categ_cols] + numeric_cols
    formula = f'{outcome_col} ~ ' + ' + '.join(parts) if parts else f'{outcome_col} ~ 1'

    model = smf.logit(formula, data=df.dropna(subset=[outcome_col] + predictor_cols))
    return model.fit(disp=False)


def fit_ordinal_regression(df: pd.DataFrame, outcome_col: str,
                            outcome_order: list, predictor_cols: list) -> object:
    """Ordinal logistic regression (proportional-odds model) via statsmodels.

    Used for:
        * J4 risk class ('High risk', 'Moderate risk', 'Low risk')
        * K1 document quality ('Opaque', 'Sparse', 'Moderate', 'Rich')

    Args:
        df: Pandas DataFrame containing the data.
        outcome_col: Name of the ordinal column containing string categories.
        outcome_order: List of target categories ordered explicitly from lowest 
            to highest (e.g., ['High risk', 'Moderate risk', 'Low risk']).
        predictor_cols: List of column names to use as features.

    Returns:
        statsmodels.miscmodels.ordinal_model.OrderedResultsWrapper: The fitted 
            ordered model results. Call `.summary()` on the returned object to print.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    df_m = df.dropna(subset=[outcome_col] + predictor_cols).copy()
    df_m[outcome_col] = pd.Categorical(df_m[outcome_col],
                                       categories=outcome_order,
                                       ordered=True)

    # Build design matrix (dummies for categoricals)
    numeric_cols = [c for c in predictor_cols
                    if pd.api.types.is_numeric_dtype(df_m[c])]
    categ_cols   = [c for c in predictor_cols
                    if not pd.api.types.is_numeric_dtype(df_m[c])]

    X_parts = []
    for c in categ_cols:
        dummies = pd.get_dummies(df_m[c], prefix=c, drop_first=True)
        X_parts.append(dummies)
    for c in numeric_cols:
        X_parts.append(df_m[[c]])

    if X_parts:
        X = pd.concat(X_parts, axis=1).astype(float)
    else:
        X = pd.DataFrame(np.ones(len(df_m)), columns=['const'])

    model = OrderedModel(df_m[outcome_col], X, distr='logit')
    return model.fit(method='bfgs', disp=False)

