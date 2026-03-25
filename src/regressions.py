'''
Regression models for predicting effect of oil price changes on stock prices.
Models :
    - OLS Linear Regression
    - AR
    - ARMA
    - VAR
    - (HAR)

Variables :
    1 - Baseline : Stock price = f(oil price)
    2 - Macro-state interaction : Stock price = f(oil price, D_macro_state) (CFNAI)
    3 - Macro-state + shock type : Stock price = f(oil price, D_macro_state, D_shock_type)

CFNAI is monthly so our period reference will be monthly. 
Daily data will be aggregated to monthly by taking the last observation of the month.
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm

def load_daily_data(
    raw_path: str = "data/raw/data_hec_project_1.xlsx",
    *,
    sheet_name: str = "Daily",
    skiprows: int = 5,
) -> pd.DataFrame:
    """Load the daily dataset from the raw Excel file.

    Notes
    -----
    In the provided Excel, the date column is named 'Dates' and some column
    names include trailing spaces (e.g., 'WTI  ').
    """
    df = pd.read_excel(raw_path, sheet_name=sheet_name, skiprows=skiprows)
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.replace(["#N/A N/A", "NA", ""], pd.NA)

    if "Date" in df.columns:
        date_col = "Date"
    elif "Dates" in df.columns:
        date_col = "Dates"
    else:
        raise KeyError(f"No date column found. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.rename(columns={date_col: "Date"})
    return df


def to_monthly_last_observation(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to monthly by taking the last observation of each month."""
    if "Date" not in daily_df.columns:
        raise ValueError("daily_df must have a 'Date' column")
    monthly = (
        daily_df.set_index("Date")
        .sort_index()
        .resample("ME")
        .last()
        .dropna(how="all")
        .reset_index()
    )
    return monthly


def prepare_monthly_sp500_wti(
    raw_path: str = "data/raw/data_hec_project_1.xlsx",
    *,
    sp_col: str = "SP500",
    wti_col: str = "WTI",
) -> pd.DataFrame:
    """Prepare monthly series for S&P500 and WTI (levels), plus WTI log-change."""
    daily = load_daily_data(raw_path)
    monthly = to_monthly_last_observation(daily)

    if sp_col not in monthly.columns:
        raise KeyError(f"Missing {sp_col!r} column. Available: {list(monthly.columns)}")
    if wti_col not in monthly.columns:
        raise KeyError(f"Missing {wti_col!r} column. Available: {list(monthly.columns)}")

    sp = pd.to_numeric(monthly[sp_col], errors="coerce")
    wti = pd.to_numeric(monthly[wti_col], errors="coerce")

    df = pd.DataFrame({"Date": monthly["Date"], sp_col: sp, wti_col: wti}).dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    df[f"{wti_col}_logdiff"] = np.log(df[wti_col]).diff()
    df = df.dropna().reset_index(drop=True)
    return df


def ols(X: pd.DataFrame, y: pd.Series, *, add_constant: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS regression wrapper."""
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit()


if __name__ == "__main__":
    df = prepare_monthly_sp500_wti()
    y = df["SP500"]
    X = df[["WTI_logdiff"]]
    model = ols(X, y)
    print(model.summary())