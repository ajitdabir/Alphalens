import math
import pandas as pd

def annualize_return_from_daily(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return 0.0
    return float(((1 + daily_returns).prod()) ** (252 / max(len(daily_returns), 1)) - 1)

def annualize_vol_from_daily(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std(ddof=0) * math.sqrt(252))

def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def tanh_score(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.tanh(x / scale)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
