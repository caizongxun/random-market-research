"""
run_pipeline.py  —  一鍵端到端流程

流程步驟
--------
  Step 1  下載並快取 OHLCV → cache/{SYM}_ohlcv.parquet
  Step 2  校準 theta       → cache/{SYM}_theta.json
  Step 3  動態 fear_threshold 分析（可選）
  Step 4  收集訓練資料     → cache/{SYM}_training_data.csv
  Step 5  訓練代理模型     → models/param_model_{SYM}.joblib
  Step 6a Rolling Forward（baseline，純 auto_calibrate）
  Step 6b Rolling Forward（+ param_model，加速版）
  Step 7  產生 HTML 對比報告 → results/pipeline/pipeline_report.html

快取邏輯：每個中間結果有 --cache-hours（預設 23）的有效期，
         重新執行時只跑過期 / 不存在的步驟。
         --force-refresh  忽略快取，全部重跑。
         --skip-steps 1 2 跳過指定步驟。
         --stop-after 5   只跑到第 5 步。

用法
----
  # 最簡單
  python scripts/run_pipeline.py --symbol AAPL

  # 多股 + 指定回測窗口
  python scripts/run_pipeline.py \\
    --symbol AAPL MSFT NVDA \\
    --end-date 2025-06-01 \\
    --total-bars 30 --step 5

  # 強制重跑
  python scripts/run_pipeline.py --symbol AAPL --force-refresh

  # 只跑到訓練（Steps 1-5）
  python scripts/run_pipeline.py --symbol AAPL --stop-after 5

  # 跳過已完成的前幾步
  python scripts/run_pipeline.py --symbol AAPL --skip-steps 1 2 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
CACHE   = ROOT / "cache"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results" / "pipeline"

CACHE.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────
def _age_hours(path: Path) -> float:
    """回傳檔案距今幾小時，不存在回傳 inf。"""
    if not path.exists():
        return float("inf")
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / 3600


def _fresh(path: Path, cache_hours: float, force: bool) -> bool:
    """True 表示快取有效（可跳過重算）。"""
    if force:
        return False
    return _age_hours(path) < cache_hours


def _run(cmd: list[str], label: str) -> int:
    """執行子程序，即時輸出，回傳 returncode。"""
    print(f"\n{'='*60}")
    print(f"  ▶  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=ROOT).returncode
    if ret != 0:
        print(f"\n[WARN] '{label}' 回傳 {ret}，繼續後續步驟。")
    return ret


def _banner(step: int, name: str):
    print(f"\n{'#'*64}")
    print(f"#  Step {step}: {name}")
    print(f"{'#'*64}")


# ──────────────────────────────────────────────────────────────
# 個別步驟
# ──────────────────────────────────────────────────────────────
def step1_download(sym: str, end_date: str | None, cache_hours: float, force: bool):
    _banner(1, f"下載並快取 OHLCV  [{sym}]")
    out = CACHE / f"{sym}_ohlcv.parquet"
    if _fresh(out, cache_hours, force):
        print(f"  [SKIP] 快取有效（{_age_hours(out):.1f} h < {cache_hours} h）→ {out}")
        return str(out)

    # 用 Python 直接下載（避免額外腳本依賴）
    try:
        import yfinance as yf
        import pandas as pd
        end_dt   = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
        start_dt = end_dt - pd.DateOffset(years=5)
        dl_end   = end_dt + pd.DateOffset(days=120)

        df = yf.download(
            sym,
            start=start_dt.strftime("%Y-%m-%d"),
            end=dl_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False, progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()
        df.to_parquet(out, index=False)
        print(f"  ✔ 已儲存 {len(df)} 根 K 棒 → {out}")
    except Exception as e:
        print(f"  [ERROR] 下載失敗：{e}")
    return str(out)


def step2_calibrate(sym: str, end_date: str | None, cache_hours: float, force: bool):
    _banner(2, f"校準 theta  [{sym}]")
    out = CACHE / f"{sym}_theta.json"
    if _fresh(out, cache_hours, force):
        print(f"  [SKIP] 快取有效 → {out}")
        return str(out)

    theta_script = SCRIPTS / "calibrate_params.py"
    if not theta_script.exists():
        print(f"  [SKIP] 找不到 {theta_script}")
        return str(out) if out.exists() else None

    cmd = [sys.executable, str(theta_script), "--symbol", sym, "--output", str(out)]
    if end_date:
        cmd += ["--end-date", end_date]
    _run(cmd, f"calibrate_params [{sym}]")
    return str(out) if out.exists() else None


def step4_collect(sym: str, theta_path: str | None,
                  end_date: str | None, step: int,
                  cache_hours: float, force: bool):
    _banner(4, f"收集訓練資料  [{sym}]")
    out = CACHE / f"{sym}_training_data.csv"
    if _fresh(out, cache_hours, force):
        print(f"  [SKIP] 快取有效 → {out}")
        return str(out)
    if theta_path is None or not Path(theta_path).exists():
        print(f"  [SKIP] 無 theta 檔案，跳過。")
        return None

    cmd = [
        sys.executable, str(SCRIPTS / "collect_training_data.py"),
        "--symbol", sym,
        "--theta",  theta_path,
        "--step",   str(step),
        "--output", str(out),
    ]
    if end_date:
        cmd += ["--end-date", end_date]
    _run(cmd, f"collect_training_data [{sym}]")
    return str(out) if out.exists() else None


def step5_train(sym: str, csv_path: str | None, cache_hours: float, force: bool):
    _banner(5, f"訓練代理模型  [{sym}]")
    out = MODELS / f"param_model_{sym}.joblib"
    if _fresh(out, cache_hours, force):
        print(f"  [SKIP] 快取有效 → {out}")
        return str(out)
    if csv_path is None or not Path(csv_path).exists():
        print(f"  [SKIP] 無訓練資料，跳過。")
        return None

    cmd = [
        sys.executable, str(SCRIPTS / "train_param_model.py"),
        "--csv",    csv_path,
        "--output", str(out),
        "--symbol", sym,
    ]
    _run(cmd, f"train_param_model [{sym}]")
    return str(out) if out.exists() else None


def step6_rolling(
    sym: str,
    theta_path: str | None,
    model_path: str | None,
    end_date: str | None,
    total_bars: int,
    step: int,
    output_dir: str,
    use_model: bool,
):
    label = f"Rolling Forward [{sym}]  {'+ param_model' if use_model else 'baseline'}"
    _banner("6b" if use_model else "6a", label)

    if theta_path is None or not Path(theta_path).exists():
        print(f"  [SKIP] 無 theta 檔案，跳過。")
        return None

    cmd = [
        sys.executable, str(SCRIPTS / "rolling_forward.py"),
        "--symbol",     sym,
        "--theta",      theta_path,
        "--total-bars", str(total_bars),
        "--step",       str(step),
        "--output-dir", output_dir,
        "--auto-calibrate",
        "--verbose",
    ]
    if end_date:
        cmd += ["--end-date", end_date]
    if use_model and model_path and Path(model_path).exists():
        cmd += ["--param-model", model_path]

    _run(cmd, label)

    # 尋找產生的 JSON
    ed = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = "_param-model" if use_model else ""
    json_path = Path(output_dir) / f"{sym}_{ed}_rolling{step}{suffix}.json"
    return str(json_path) if json_path.exists() else None


def step7_report(
    sym_results: dict,   # {sym: {"baseline": json_path, "model": json_path}}
    output_dir: str,
):
    _banner(7, "產生 HTML 對比報告")

    rows_html = ""
    for sym, paths in sym_results.items():
        base_j = paths.get("baseline")
        modl_j = paths.get("model")

        def _load(p):
            if p and Path(p).exists():
                with open(p) as f:
                    return json.load(f)
            return {}

        base = _load(base_j)
        modl = _load(modl_j)

        def _fmt(d, key, fmt=".2f"):
            v = d.get(key)
            return f"{v:{fmt}}" if v is not None else "N/A"

        def _color(base_v, model_v):
            """模型 MAE 比 baseline 小 → 綠，否則 → 紅。"""
            if base_v is None or model_v is None:
                return ""
            return "color:#27ae60;font-weight:bold" if model_v < base_v else "color:#e74c3c;font-weight:bold"

        b_mae  = base.get("overall_mae_pct")
        m_mae  = modl.get("overall_mae_pct")
        b_dir  = base.get("dir_accuracy_pct")
        m_dir  = modl.get("dir_accuracy_pct")

        mae_style = _color(b_mae, m_mae)
        # 方向命中：模型高 → 綠
        dir_style = _color(
            100 - b_dir if b_dir is not None else None,
            100 - m_dir if m_dir is not None else None,
        )

        rows_html += f"""
        <tr>
          <td><b>{sym}</b></td>
          <td>{_fmt(base, 'overall_mae_pct')}%</td>
          <td style="{mae_style}">{_fmt(modl, 'overall_mae_pct')}%</td>
          <td>{_fmt(base, 'dir_accuracy_pct')}%</td>
          <td style="{dir_style}">{_fmt(modl, 'dir_accuracy_pct')}%</td>
          <td>{base.get('total_bars', 'N/A')}</td>
          <td>{base.get('step', 'N/A')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>Pipeline Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#f5f6fa; margin:0; padding:24px; }}
  h1   {{ color:#2c3e50; font-size:1.6rem; margin-bottom:8px; }}
  .meta{{ color:#777; font-size:0.85rem; margin-bottom:24px; }}
  table{{ border-collapse:collapse; width:100%; background:#fff;
          box-shadow:0 2px 8px rgba(0,0,0,.08); border-radius:8px; overflow:hidden; }}
  th   {{ background:#2c3e50; color:#fff; padding:10px 14px; text-align:left; font-size:0.9rem; }}
  td   {{ padding:9px 14px; border-bottom:1px solid #eee; font-size:0.88rem; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f0f4ff; }}
  .legend{{ margin-top:16px; font-size:0.82rem; color:#555; }}
  .green{{ color:#27ae60; font-weight:bold; }}
  .red  {{ color:#e74c3c; font-weight:bold; }}
</style>
</head>
<body>
<h1>📊 Pipeline Report</h1>
<div class="meta">生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<table>
  <thead>
    <tr>
      <th>Symbol</th>
      <th>Baseline MAE%</th>
      <th>Model MAE%</th>
      <th>Baseline Dir%</th>
      <th>Model Dir%</th>
      <th>Total Bars</th>
      <th>Step</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
<div class="legend">
  <span class="green">■ 綠色</span>：Model 優於 Baseline　
  <span class="red">■ 紅色</span>：Model 劣於 Baseline
</div>
</body>
</html>"""

    out = Path(output_dir) / "pipeline_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  ✔ HTML 報告 → {out}")
    return str(out)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline")
    p.add_argument("--symbol",       nargs="+", required=True, help="股票代號，可多個")
    p.add_argument("--end-date",     default=None,  help="預測起點截止日 YYYY-MM-DD")
    p.add_argument("--total-bars",   type=int, default=30)
    p.add_argument("--step",         type=int, default=5)
    p.add_argument("--cache-hours",  type=float, default=23.0,
                   help="快取有效時間（小時），預設 23")
    p.add_argument("--force-refresh",action="store_true",
                   help="忽略快取，全部重跑")
    p.add_argument("--skip-steps",   nargs="+", type=int, default=[],
                   help="跳過指定步驟編號，如 --skip-steps 1 2 3")
    p.add_argument("--stop-after",   type=int, default=7,
                   help="跑完哪一步後停止（預設 7 = 全部）")
    p.add_argument("--output-dir",   default=str(RESULTS))
    return p.parse_args()


def main():
    args = parse_args()
    skip  = set(args.skip_steps)
    stop  = args.stop_after

    sym_results: dict = {}

    for sym in args.symbol:
        sym = sym.upper()
        print(f"\n{'*'*64}")
        print(f"*  處理股票：{sym}")
        print(f"{'*'*64}")

        theta_path = None
        csv_path   = None
        model_path = None

        # Step 1
        if 1 not in skip and stop >= 1:
            step1_download(sym, args.end_date, args.cache_hours, args.force_refresh)

        # Step 2
        if 2 not in skip and stop >= 2:
            theta_path = step2_calibrate(sym, args.end_date, args.cache_hours, args.force_refresh)
        else:
            # 嘗試從快取讀取
            candidate = CACHE / f"{sym}_theta.json"
            theta_path = str(candidate) if candidate.exists() else None

        # Step 3：fear_threshold 分析（可選，若腳本存在才執行）
        if 3 not in skip and stop >= 3:
            ft_script = SCRIPTS / "analyze_fear_threshold.py"
            if ft_script.exists() and theta_path and Path(theta_path).exists():
                _banner(3, f"Fear Threshold 分析  [{sym}]")
                ft_out = CACHE / f"{sym}_fear_profile.json"
                if not _fresh(ft_out, args.cache_hours, args.force_refresh):
                    cmd = [
                        sys.executable, str(ft_script),
                        "--symbol", sym,
                        "--theta",  theta_path,
                        "--output", str(ft_out),
                    ]
                    if args.end_date:
                        cmd += ["--end-date", args.end_date]
                    _run(cmd, f"analyze_fear_threshold [{sym}]")
                else:
                    print(f"  [SKIP] Fear profile 快取有效 → {ft_out}")

        # Step 4
        if 4 not in skip and stop >= 4:
            csv_path = step4_collect(
                sym, theta_path, args.end_date,
                args.step, args.cache_hours, args.force_refresh,
            )
        else:
            candidate = CACHE / f"{sym}_training_data.csv"
            csv_path = str(candidate) if candidate.exists() else None

        # Step 5
        if 5 not in skip and stop >= 5:
            model_path = step5_train(sym, csv_path, args.cache_hours, args.force_refresh)
        else:
            candidate = MODELS / f"param_model_{sym}.joblib"
            model_path = str(candidate) if candidate.exists() else None

        base_json  = None
        model_json = None

        # Step 6a  baseline
        if 6 not in skip and stop >= 6:
            base_json = step6_rolling(
                sym, theta_path, None,
                args.end_date, args.total_bars, args.step,
                args.output_dir, use_model=False,
            )

        # Step 6b  param-model
        if 6 not in skip and stop >= 6:
            model_json = step6_rolling(
                sym, theta_path, model_path,
                args.end_date, args.total_bars, args.step,
                args.output_dir, use_model=True,
            )

        sym_results[sym] = {"baseline": base_json, "model": model_json}

    # Step 7
    if 7 not in skip and stop >= 7:
        report_path = step7_report(sym_results, args.output_dir)
        print(f"\n{'='*64}")
        print(f"  ✅ Pipeline 完成！報告：{report_path}")
        print(f"{'='*64}\n")
    else:
        print(f"\n{'='*64}")
        print(f"  ✅ Pipeline 完成（步驟 ≤{stop}）")
        print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
