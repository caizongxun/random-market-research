"""
calibrate_agent_profile.py

從歷史 OHLCV 校準代理人行為參數，輸出 cache/{SYM}_agent_profile.json

代理人分層：
  散戶層（~90%）
    - momentum_follower : 近期漲就買、跌就賣，強度 = momentum_strength_retail
    - stop_loss         : 跌破 stop_loss_threshold 觸發賣壓
    - holder            : 套牢不動，在原買入價形成阻力（用 volume profile 估計）
    - panic             : VIX 高時拋售機率上升，以 fear_sensitivity 量化

  大戶層（~10%）
    - mean_reversion    : 偏離 backbone 越遠，回拉力越強，強度 = inst_mr_strength
    - volume_absorption : 成交量放大 + 低位 → 吸籌，量化為 absorption_vol_ratio
    - distribution      : 接近前高阻力位時出貨壓力，以 dist_resist_zone_pct 量化

輸出 JSON 欄位：
    retail_momentum_strength  : 散戶動能強度（0~1）
    retail_stop_loss_threshold: 觸發停損的跌幅閾值（%）
    retail_panic_sensitivity  : 恐慌拋售靈敏度（0~1）
    inst_mr_strength          : 大戶均值回歸強度（0~1）
    inst_absorption_ratio     : 大戶吸籌量比（1.0 = 正常）
    inst_dist_zone_pct        : 大戶出貨壓力區寬度（%）
    vix_level                 : 最近 VIX 均值（若可取得）
    calibration_bars          : 校準用的 K 棒數
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def _fetch_vix(start: str, end: str) -> float | None:
    """嘗試抓 VIX，失敗回傳 None。"""
    try:
        vix = yf.download("^VIX", start=start, end=end, interval="1d",
                          auto_adjust=False, progress=False)
        if len(vix) == 0:
            return None
        close = vix["Close"].values.astype(float)
        return float(np.mean(close[-20:]))
    except Exception:
        return None


def calibrate_retail_momentum(log_rets: np.ndarray, window: int = 5) -> float:
    """
    散戶動能強度：用 short-window autocorrelation 估計。
    autocorr > 0 → 趨勢追隨較強；autocorr < 0 → 反向較強。
    回傳 0~1，0.5 為中性。
    """
    s = pd.Series(log_rets)
    ac = s.autocorr(lag=window)
    if np.isnan(ac):
        return 0.5
    # 映射 [-1, 1] → [0, 1]
    return float(np.clip((ac + 1) / 2, 0.0, 1.0))


def calibrate_stop_loss_threshold(log_rets: np.ndarray, percentile: float = 10) -> float:
    """
    停損閾值：取 log_rets 的 P10（最壞跌幅的前 10%），轉為絕對百分比。
    代表散戶在這個跌幅下容易觸發停損。
    """
    p = float(np.percentile(log_rets, percentile))
    return float(abs(p) * 100)  # 轉為 %


def calibrate_panic_sensitivity(log_rets: np.ndarray, vix: float | None) -> float:
    """
    恐慌靈敏度：結合波動率分佈的肥尾程度 + VIX 水平。
    VIX > 30 → 高恐慌；VIX < 15 → 低恐慌。
    無 VIX 時純用 kurtosis 估計。
    """
    try:
        from scipy.stats import kurtosis
        k = float(kurtosis(log_rets, fisher=True))
    except Exception:
        k = 0.0

    # kurtosis 貢獻：常態=0，肥尾>0，最大貢獻 0.3
    kurt_score = float(np.clip(k / 10, 0.0, 0.3))

    # VIX 貢獻：佔 0.7
    if vix is not None:
        vix_score = float(np.clip((vix - 15) / 30, 0.0, 0.7))
    else:
        # 無 VIX，用波動率水平估計
        vol = float(np.std(log_rets) * np.sqrt(252))
        vix_score = float(np.clip((vol - 0.10) / 0.40, 0.0, 0.7))

    return float(np.clip(kurt_score + vix_score, 0.0, 1.0))


def calibrate_inst_mr_strength(
    close: np.ndarray,
    window: int = 20,
) -> float:
    """
    大戶均值回歸強度：
    用 Hurst exponent（R/S 法）估計。
    H < 0.5 → 均值回歸傾向強 → inst_mr_strength 高。
    H > 0.5 → 趨勢傾向強。
    回傳 0~1，0.5 為中性。
    """
    log_c = np.log(close)
    n = len(log_c)
    if n < 40:
        return 0.5

    # 簡單 R/S 估計
    splits = [n // 4, n // 2, n]
    rs_vals = []
    for s in splits:
        seg = log_c[:s]
        mean = np.mean(seg)
        dev = np.cumsum(seg - mean)
        r = np.max(dev) - np.min(dev)
        std = np.std(seg)
        if std > 0:
            rs_vals.append((np.log(s), np.log(r / std)))

    if len(rs_vals) < 2:
        return 0.5

    xs = np.array([v[0] for v in rs_vals])
    ys = np.array([v[1] for v in rs_vals])
    H = float(np.polyfit(xs, ys, 1)[0])
    H = float(np.clip(H, 0.0, 1.0))

    # H < 0.5 → 回歸強，映射到較高 mr_strength
    mr_strength = float(np.clip(1.0 - H, 0.0, 1.0))
    return mr_strength


def calibrate_absorption_ratio(
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> float:
    """
    大戶吸籌比率：
    在下跌段中，若成交量相對均值放大，表示大戶可能在承接。
    回傳比率（1.0 = 正常，> 1 = 吸籌訊號明顯）。
    """
    log_rets = np.diff(np.log(close))
    vol_ma = pd.Series(volume[1:]).rolling(window).mean().values

    down_mask = log_rets < 0
    if down_mask.sum() == 0:
        return 1.0

    down_vol = volume[1:][down_mask]
    down_ma  = vol_ma[down_mask]
    valid    = down_ma > 0
    if valid.sum() == 0:
        return 1.0

    ratio = float(np.mean(down_vol[valid] / down_ma[valid]))
    return float(np.clip(ratio, 0.5, 3.0))


def calibrate_dist_zone(
    close: np.ndarray,
    lookback: int = 120,
) -> float:
    """
    大戶出貨壓力區：
    估計前高阻力附近的價格密度，回傳阻力帶寬度（%）。
    """
    c = close[-lookback:] if len(close) > lookback else close
    high = float(np.max(c))
    # 取前高附近 5% 以內的價格，計算密度
    zone_mask = c >= high * 0.95
    if zone_mask.sum() < 2:
        return 3.0  # 預設 3%
    zone_prices = c[zone_mask]
    spread = float((np.max(zone_prices) - np.min(zone_prices)) / np.mean(zone_prices) * 100)
    return float(np.clip(spread, 1.0, 10.0))


def calibrate_agent_profile(
    df: pd.DataFrame,
    symbol: str,
    window: int = 500,
) -> dict:
    """主校準函數，回傳 agent_profile dict。"""
    df_w = df.iloc[-window:] if len(df) > window else df
    close  = df_w["Close"].values.astype(float)
    volume = df_w["Volume"].values.astype(float)
    log_rets = np.diff(np.log(close))

    # 嘗試取 VIX
    start_str = str(df_w.index[0].date()) if hasattr(df_w.index[0], "date") else None
    end_str   = str(df_w.index[-1].date()) if hasattr(df_w.index[-1], "date") else None
    vix = _fetch_vix(start_str, end_str) if (start_str and end_str) else None
    print(f"  VIX 均值: {vix:.1f}" if vix else "  VIX: 無法取得，使用波動率估計")

    retail_momentum  = calibrate_retail_momentum(log_rets)
    stop_loss_thresh = calibrate_stop_loss_threshold(log_rets)
    panic_sens       = calibrate_panic_sensitivity(log_rets, vix)
    inst_mr          = calibrate_inst_mr_strength(close)
    absorption       = calibrate_absorption_ratio(close, volume)
    dist_zone        = calibrate_dist_zone(close)

    profile = {
        "symbol":                      symbol,
        "retail_momentum_strength":    round(retail_momentum,  4),
        "retail_stop_loss_threshold":  round(stop_loss_thresh, 4),
        "retail_panic_sensitivity":    round(panic_sens,       4),
        "inst_mr_strength":            round(inst_mr,          4),
        "inst_absorption_ratio":       round(absorption,       4),
        "inst_dist_zone_pct":          round(dist_zone,        4),
        "vix_level":                   round(vix, 2) if vix else None,
        "calibration_bars":            len(df_w),
    }

    print(f"  散戶動能強度    : {retail_momentum:.3f}")
    print(f"  停損閾值        : {stop_loss_thresh:.3f}%")
    print(f"  恐慌靈敏度      : {panic_sens:.3f}")
    print(f"  大戶均值回歸強度: {inst_mr:.3f}")
    print(f"  大戶吸籌比率    : {absorption:.3f}")
    print(f"  出貨壓力區寬度  : {dist_zone:.3f}%")

    return profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--end-date",  default=None)
    p.add_argument("--window",    type=int, default=500)
    p.add_argument("--cache-dir", default="cache")
    args = p.parse_args()

    end_dt   = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today()
    start_dt = end_dt - pd.DateOffset(years=3)

    print(f"下載 {args.symbol} {start_dt.date()} ~ {end_dt.date()} ...")
    raw = yf.download(
        args.symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    for col in ["Date", "Datetime"]:
        if col in raw.columns:
            raw = raw.set_index(col)
            break
    raw.index = pd.to_datetime(raw.index)
    df = raw[raw.index <= end_dt].copy()
    print(f"共 {len(df)} 根 K 棒")

    print(f"\n校準代理人行為參數 ({args.symbol}) ...")
    profile = calibrate_agent_profile(df, args.symbol, window=args.window)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{args.symbol}_agent_profile.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"\n✔ agent_profile → {out_path}")


if __name__ == "__main__":
    main()
