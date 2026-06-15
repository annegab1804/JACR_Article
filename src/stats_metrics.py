import math
import numpy as np
import pandas as pd

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score confidence interval for a proportion.

    The Wilson interval performs far better than the normal approximation near 
    0% and 100%, which is common in this dataset (e.g., race reporting). 
    Required by PDF spec for every reporting-rate proportion.

    Args:
        k: Number of successes.
        n: Total number of observations.
        z: Critical value. Defaults to 1.96 (for a 95% confidence interval). 
            Use 2.576 for a 99% confidence interval.

    Returns:
        tuple: A tuple of two floats `(lower, upper)` representing the confidence 
            interval bounds. Returns `(np.nan, np.nan)` if `n == 0`.
    """
    if n == 0:
        return (np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

def reporting_rate_with_ci(series: pd.Series, positive_check) -> dict:
    """Compute a reporting rate and its 95% Wilson CI for a column.

    Args:
        series: Pandas Series containing the raw column values to analyze.
        positive_check: A callable (function or lambda) that accepts a value 
            and returns True if the value counts as 'reported'.

    Returns:
        dict: A dictionary containing the computed metrics with the following keys:
            * 'rate' (float): The calculated reporting proportion (k / n).
            * 'ci_low' (float): Lower bound of the 95% Wilson score interval.
            * 'ci_high' (float): Upper bound of the 95% Wilson score interval.
            * 'n' (int): Total number of rows/observations in the series.
            * 'k' (int): Count of rows meeting the positive criteria.
    """
    n = len(series)
    k = int(series.apply(positive_check).sum())
    rate = k / n if n > 0 else np.nan
    ci_low, ci_high = wilson_ci(k, n)
    return {'rate': rate, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n, 'k': k}

def herfindahl_hirschman_index(counts) -> float:
    """Herfindahl–Hirschman Index on a sequence of counts (or a pd.Series).

    Formula used: HHI = Σ (share_i)² × 10,000.
    Values greater than 2,500 indicate high concentration (monopoly = 10,000).

    Args:
        counts: An array-like sequence or pandas Series containing raw counts 
            (not shares). Zeros are automatically ignored.

    Returns:
        float: The computed HHI score, or `np.nan` if the sum of all counts 
            is equal to zero.
    """
    arr = np.array([c for c in counts if c > 0], dtype=np.float64)
    if arr.sum() == 0:
        return np.nan
    shares = arr / arr.sum()
    return float((shares ** 2).sum() * 10_000)

def shannon_diversity(counts) -> float:
    """Shannon diversity index H = −Σ p_i × ln(p_i).

    Args:
        counts: An array-like sequence containing raw counts. Zeros are 
            excluded from the calculation (0 × ln 0 → 0 by convention).

    Returns:
        float: The calculated Shannon index H, or `np.nan` if all counts 
            evaluate to zero.
    """
    arr = np.array([c for c in counts if c > 0], dtype=np.float64)
    if arr.sum() == 0:
        return np.nan
    p = arr / arr.sum()
    return float(-np.sum(p * np.log(p)))

def calculate_gini(array) -> float:
    """Gini coefficient of inequality from raw value frequencies.

    Args:
        array: An array-like sequence of raw frequencies or values.

    Returns:
        float: A value between 0.0 and 1.0, where 0 represents perfect equality 
            and 1 represents maximum inequality. Returns `np.nan` if the input 
            array is empty or sums to zero.
    """
    arr = np.array(array, dtype=np.float64)
    if len(arr) == 0 or arr.sum() == 0:
        return np.nan
    arr = np.sort(arr)
    index = np.arange(1, len(arr) + 1)
    n = len(arr)
    return float(((2 * index - n - 1) * arr).sum() / (n * arr.sum()))

def lorenz_curve(counts) -> tuple:
    """Compute Lorenz curve coordinates from a counts array.

    Args:
        counts: An array-like sequence of raw frequencies or counts.

    Returns:
        tuple: A tuple containing two 1D NumPy arrays `(x, y)`:
            * x (np.ndarray): Cumulative share of countries, ranging from 0.0 to 1.0.
            * y (np.ndarray): Cumulative share of devices, ranging from 0.0 to 1.0.
    """
    arr = np.sort(np.array(counts, dtype=np.float64))
    cumulative = np.cumsum(arr)
    x = np.linspace(0, 1, len(arr) + 1)
    y = np.concatenate([[0], cumulative / cumulative[-1]])
    return x, y

def cochran_armitage_trend_test(years: pd.Series, reported: pd.Series) -> dict:
    """Cochran–Armitage trend test for linear changes in binary flags over time.

    Determines whether a binary reporting flag has changed linearly over 
    the decision years. Uses the large-sample normal approximation (suitable 
    for n > 30). Required by PDF spec for year-over-year demographic-reporting 
    trends.

    Args:
        years: Pandas Series representing the numeric year (from Decision_year).
        reported: Pandas Series containing binary 0/1 values, where 1 indicates 
            the field was successfully reported.

    Returns:
        dict: A dictionary containing the test outputs with the following keys:
            * 'T_statistic' (float): The calculated Z/T statistic score.
            * 'p_value' (float): The two-sided p-value from the normal distribution.
            * 'n' (int): Total number of valid, non-null observations used.
    """
    df_t = pd.DataFrame({'year': pd.to_numeric(years, errors='coerce'),
                         'y': pd.to_numeric(reported, errors='coerce')}).dropna()
    n = len(df_t)
    if n < 2:
        return {'T_statistic': np.nan, 'p_value': np.nan, 'n': n}

    x = df_t['year'].values
    y = df_t['y'].values
    x_bar = x.mean()
    p_bar = y.mean()

    T = np.sum((x - x_bar) * y)
    var_T = p_bar * (1 - p_bar) * np.sum((x - x_bar) ** 2)
    if var_T == 0:
        return {'T_statistic': np.nan, 'p_value': np.nan, 'n': n}

    from scipy import stats as _stats
    z = T / math.sqrt(var_T)
    p_value = 2 * _stats.norm.sf(abs(z))
    return {'T_statistic': float(z), 'p_value': float(p_value), 'n': n}

