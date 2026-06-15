"""
app.py  --  Streamlit Web Dashboard

啟動：  streamlit run app.py

功能
----
  Run Pipeline : 在網頁上觸發完整 7 步流程（spawn subprocess）
  Pipeline Results : 讀取 pipeline_report.html 內嵌顯示
  MAE / Direction / Coverage Chart : Plotly 圖表對比 Baseline vs Model
  Cache Status : 檢查各快叔檔案年齡
  Rolling Result Viewer : 自選檔案預覽 JSON rolling 詳情
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT     = Path(__file__).parent
CACHE    = ROOT / "cache"
MODELS   = ROOT / "models"
RESULTS  = ROOT / "results" / "pipeline"
SCRIPTS  = ROOT / "scripts"

st.set_page_config(
    page_title="Market Simulation Pipeline",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Market Simulation")
    st.markdown("**OU + GJR-GARCH** | Rolling Forward | Surrogate Model")
    st.divider()

    symbols_raw = st.text_input("Symbols（空格分隔）", value="AAPL")
    symbols = [s.strip().upper() for s in symbols_raw.split() if s.strip()]

    end_date = st.text_input("End Date（YYYY-MM-DD）", value="", placeholder="留空 = 今日")
    end_date_arg = end_date.strip() or None

    total_bars = st.slider("Total Bars（Rolling 筆數）", 10, 120, 30)
    step       = st.slider("Step（每次捲動）",           1,  30,  5)

    st.divider()
    force    = st.checkbox("強制重跑（--force-refresh）", value=False)
    stop_at  = st.selectbox("止於 Step", options=[7,1,2,3,4,5,6], index=0,
                            format_func=lambda x: f"Step {x}")

    run_btn  = st.button("🚀  執行 Pipeline", use_container_width=True, type="primary")
    st.divider()
    tab_sel  = st.radio(
        "瀏覽",
        ["Pipeline 報告", "MAE / Dir / Coverage 圖",
         "Cache 狀態", "Rolling JSON 詳情"],
        index=0,
    )


# ──────────────────────────────────────────────────────────────
# 執行 Pipeline
# ──────────────────────────────────────────────────────────────
if run_btn:
    if not symbols:
        st.warning("請輸入至少一個股票代號")
    else:
        cmd = [
            sys.executable, str(SCRIPTS / "run_pipeline.py"),
            "--symbol", *symbols,
            "--total-bars", str(total_bars),
            "--step",        str(step),
            "--stop-after",  str(stop_at),
        ]
        if end_date_arg:
            cmd += ["--end-date", end_date_arg]
        if force:
            cmd += ["--force-refresh"]

        st.info(f"CMD: `{' '.join(cmd)}`")
        log_area = st.empty()
        log_lines: list[str] = []
        t0 = time.time()
        with st.spinner("執行中，請稍候..."):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=ROOT
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-60:]), language="bash")
            proc.wait()

        elapsed = time.time() - t0
        if proc.returncode == 0:
            st.success(f"✅ Pipeline 完成！耗時 {elapsed:.1f}s")
        else:
            st.error(f"❌ returncode={proc.returncode}")

        st.code("\n".join(log_lines), language="bash")


# ──────────────────────────────────────────────────────────────
# Tab: Pipeline 報告
# ──────────────────────────────────────────────────────────────
if tab_sel == "Pipeline 報告":
    st.header("📄 Pipeline Report")
    html_path = RESULTS / "pipeline_report.html"
    if html_path.exists():
        st.components.v1.html(
            html_path.read_text(encoding="utf-8"),
            height=600, scrolling=True,
        )
    else:
        st.info("報告尚未產生，請先執行 Pipeline。")


# ──────────────────────────────────────────────────────────────
# Tab: MAE / Dir / Coverage 圖
# ──────────────────────────────────────────────────────────────
elif tab_sel == "MAE / Dir / Coverage 圖":
    import plotly.graph_objects as go

    st.header("📊 Baseline vs Model 指標對比")

    # 收集所有 rolling JSON
    json_files = sorted(RESULTS.glob("*.json"))
    if not json_files:
        st.info("尚無 JSON 結果，請先執行 Pipeline。")
    else:
        rows = []
        for jf in json_files:
            try:
                d = json.loads(jf.read_text())
                # 從檔名推断張號和類型
                stem = jf.stem  # e.g. AAPL_2025-06-01_rolling5_medoid
                parts = stem.split("_")
                sym  = parts[0] if parts else jf.stem
                kind = "model" if ("agent" in stem or "param" in stem) else "baseline"
                rows.append({
                    "sym":  sym,
                    "kind": kind,
                    "mae":  d.get("overall_mae_pct"),
                    "dir":  d.get("dir_accuracy_pct"),
                    "cov":  d.get("coverage_p10_p90"),
                    "file": jf.name,
                })
            except Exception:
                pass

        if not rows:
            st.warning("最新執行的 JSON 中無法解析指標。")
        else:
            df = pd.DataFrame(rows).dropna(subset=["mae"])
            syms = df["sym"].unique().tolist()

            for metric, label, higher_better in [
                ("mae", "MAE% （越低越好）", False),
                ("dir", "Direction Accuracy% （越高越好）", True),
                ("cov", "Coverage P10-P90% （越高越好）", True),
            ]:
                sub = df.dropna(subset=[metric])
                if sub.empty:
                    continue
                fig = go.Figure()
                colors = {"baseline": "#5b8dee", "model": "#27ae60"}
                for kind in ["baseline", "model"]:
                    kdf = sub[sub["kind"] == kind]
                    if kdf.empty:
                        continue
                    fig.add_trace(go.Bar(
                        name=kind.capitalize(),
                        x=kdf["sym"],
                        y=kdf[metric],
                        marker_color=colors[kind],
                        text=[f"{v:.2f}" for v in kdf[metric]],
                        textposition="outside",
                    ))
                fig.update_layout(
                    title=label,
                    barmode="group",
                    plot_bgcolor="#f5f6fa",
                    paper_bgcolor="#f5f6fa",
                    height=380,
                    margin=dict(t=40, b=30, l=30, r=10),
                    legend=dict(orientation="h", y=1.1),
                )
                if metric == "mae":
                    fig.add_hline(y=2.0, line_dash="dot", line_color="#e74c3c",
                                  annotation_text="2% 參考線")
                elif metric == "cov":
                    fig.add_hline(y=80.0, line_dash="dot", line_color="#8e44ad",
                                  annotation_text="80% 目標")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.dataframe(
                df[["sym", "kind", "mae", "dir", "cov", "file"]]
                  .rename(columns={"sym": "Symbol", "kind": "Type",
                                    "mae": "MAE%", "dir": "Dir%",
                                    "cov": "Cov%", "file": "Source"})
                  .sort_values(["Symbol", "Type"]),
                use_container_width=True,
                hide_index=True,
            )


# ──────────────────────────────────────────────────────────────
# Tab: Cache 狀態
# ──────────────────────────────────────────────────────────────
elif tab_sel == "Cache 狀態":
    st.header("🗂️ Cache 檔案年齡")

    def _age(p: Path) -> str:
        if not p.exists():
            return "✖．不存在"
        h = (time.time() - p.stat().st_mtime) / 3600
        if h < 1:
            return f"✅ {h*60:.0f} min"
        return f"✅ {h:.1f} h"

    rows = []
    # OHLCV
    for sym in (symbols if symbols else ["AAPL"]):
        rows.append({"File": f"{sym}_ohlcv.parquet",     "Status": _age(CACHE / f"{sym}_ohlcv.parquet")})
        rows.append({"File": f"{sym}_theta.json",        "Status": _age(CACHE / f"{sym}_theta.json")})
        rows.append({"File": f"{sym}_agent_profile.json","Status": _age(CACHE / f"{sym}_agent_profile.json")})
        rows.append({"File": f"{sym}_training_data.csv", "Status": _age(CACHE / f"{sym}_training_data.csv")})
        rows.append({"File": f"param_model_{sym}.joblib", "Status": _age(MODELS / f"param_model_{sym}.joblib")})
    rows.append({"File": "macro.parquet", "Status": _age(CACHE / "macro.parquet")})

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ──────────────────────────────────────────────────────────────
# Tab: Rolling JSON 詳情
# ──────────────────────────────────────────────────────────────
elif tab_sel == "Rolling JSON 詳情":
    st.header("🔎 Rolling Result 檔案詳情")
    json_files = sorted(RESULTS.glob("*.json"))
    if not json_files:
        st.info("尚無 JSON 檔案。")
    else:
        selected = st.selectbox(
            "選擇檔案",
            options=json_files,
            format_func=lambda p: p.name,
        )
        if selected:
            data = json.loads(Path(selected).read_text())
            st.subheader(從檔名 = f"📄 {selected.name}")
            # 高標 + per_bar 分離顯示
            summary_keys = [k for k in data if not isinstance(data[k], list)]
            detail_keys  = [k for k in data if isinstance(data[k], list)]

            st.markdown("**Summary**")
            st.json({k: data[k] for k in summary_keys})

            for dk in detail_keys:
                with st.expander(f"{dk} ({len(data[dk])} rows)"):
                    st.dataframe(
                        pd.DataFrame(data[dk]) if isinstance(data[dk][0], dict)
                        else pd.Series(data[dk], name=dk).to_frame(),
                        use_container_width=True,
                    )
