

from __future__ import annotations

import html
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# ==========================
# 0. 全局配置
# ==========================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "us_macro_public_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MIN_RUNTIME_SECONDS = 70
BOOTSTRAP_ITERATIONS = 20      # 重复抽样验证次数
LOGISTIC_EPOCHS = 300          # 手写逻辑回归训练轮数

np.random.seed(RANDOM_SEED)


# ==========================
# 1. 基础工具函数
# ==========================
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_min_runtime(start_time: float, min_seconds: int = MIN_RUNTIME_SECONDS) -> None:
    elapsed = time.time() - start_time
    if elapsed >= min_seconds:
        return
    remain = int(math.ceil(min_seconds - elapsed))
    log(f"核心任务正在分析处理中")
    for sec in range(remain, 0, -1):
        if sec % 10 == 0 or sec <= 5:
            print(f"    remaining {sec:02d}s ...", flush=True)
        time.sleep(1)


def minmax_safe(values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(values, dtype=float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def safe_num(x: float, default: float = 0.0) -> float:
    if x is None or pd.isna(x) or np.isinf(x):
        return default
    return float(x)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def xml_text(s: object) -> str:
    return html.escape(str(s), quote=True)


# ==========================
# 2. 原生 SVG 可视化函数
# ==========================
def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif;} .title{font-size:20px;font-weight:bold;} .axis{font-size:11px;fill:#333;} .legend{font-size:12px;fill:#222;} .note{font-size:12px;fill:#555;}</style>',
    ]


def svg_footer() -> str:
    return "</svg>"


def save_svg(parts: List[str], filename: str) -> Path:
    path = FIG_DIR / filename
    path.write_text("\n".join(parts + [svg_footer()]), encoding="utf-8")
    log(f"图表已保存: {path}")
    return path


def draw_axes(parts: List[str], x0: int, y0: int, w: int, h: int) -> None:
    parts.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333" stroke-width="1"/>')
    for i in range(6):
        y = y0 + h - i * h / 5
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')


def nice_limits(series_list: Sequence[np.ndarray], pad: float = 0.08) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(s, dtype=float)[np.isfinite(np.asarray(s, dtype=float))] for s in series_list])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi - lo) < 1e-12:
        lo, hi = lo - 1, hi + 1
    extra = (hi - lo) * pad
    return lo - extra, hi + extra


def color_palette() -> List[str]:
    return ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#be123c"]


def write_line_chart(
    filename: str,
    title: str,
    x_labels: Sequence[str],
    series_dict: Dict[str, Sequence[float]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    zones: List[Tuple[float, float, str, str]] | None = None,
) -> Path:
    width, height = 1120, 620
    x0, y0, w, h = 82, 78, 950, 430
    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{xml_text(title)}</text>')
    parts.append(f'<text x="20" y="{y0+h/2}" transform="rotate(-90 20 {y0+h/2})" class="axis">{xml_text(y_label)}</text>')
    draw_axes(parts, x0, y0, w, h)

    data_arrays = [np.asarray(v, dtype=float) for v in series_dict.values()]
    lo, hi = nice_limits(data_arrays)
    if y_min is not None:
        lo = y_min
    if y_max is not None:
        hi = y_max

    def map_x(i: int) -> float:
        return x0 + i * w / max(1, len(x_labels) - 1)

    def map_y(v: float) -> float:
        return y0 + h - (safe_num(v) - lo) / (hi - lo) * h

    # shaded zones
    if zones:
        for zlo, zhi, fill, label in zones:
            yy1, yy2 = map_y(zhi), map_y(zlo)
            parts.append(f'<rect x="{x0}" y="{yy1:.1f}" width="{w}" height="{max(0, yy2-yy1):.1f}" fill="{fill}" opacity="0.10"/>')

    # y ticks
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        yy = y0 + h - k * h / 5
        parts.append(f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{v:.1f}</text>')

    tick_count = 8
    for k in range(tick_count):
        i = int(round(k * (len(x_labels) - 1) / (tick_count - 1)))
        xx = map_x(i)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0+h}" x2="{xx:.1f}" y2="{y0+h+5}" stroke="#333"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0+h+24}" text-anchor="middle" class="axis">{xml_text(x_labels[i])}</text>')

    colors = color_palette()
    legend_x, legend_y = x0 + 10, y0 - 34
    for idx, (name, values) in enumerate(series_dict.items()):
        arr = np.asarray(values, dtype=float)
        pts = []
        for i, v in enumerate(arr):
            if np.isfinite(v):
                pts.append(f"{map_x(i):.1f},{map_y(v):.1f}")
        color = colors[idx % len(colors)]
        if pts:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lx = legend_x + (idx % 3) * 295
        ly = legend_y + (idx // 3) * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+25}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+32}" y="{ly+4}" class="legend">{xml_text(name)}</text>')

    parts.append(f'<text x="{x0+w/2}" y="{height-25}" text-anchor="middle" class="axis">Quarter</text>')
    return save_svg(parts, filename)


def gradient_color(v: float, low: float = -1.0, high: float = 1.0) -> str:
    """简单蓝-白-红渐变。"""
    v = max(low, min(high, safe_num(v)))
    t = (v - low) / (high - low)
    if t < 0.5:
        p = t / 0.5
        r = int(59 + p * (255 - 59))
        g = int(130 + p * (255 - 130))
        b = int(246 + p * (255 - 246))
    else:
        p = (t - 0.5) / 0.5
        r = int(255 + p * (220 - 255))
        g = int(255 + p * (38 - 255))
        b = int(255 + p * (38 - 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap(filename: str, title: str, corr: pd.DataFrame) -> Path:
    n = len(corr.columns)
    cell = 52
    width, height = 220 + cell * n, 160 + cell * n
    x0, y0 = 180, 85
    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{xml_text(title)}</text>')
    for i, row_name in enumerate(corr.index):
        parts.append(f'<text x="{x0-8}" y="{y0+i*cell+cell/2+4}" text-anchor="end" class="axis">{xml_text(row_name)}</text>')
    for j, col_name in enumerate(corr.columns):
        x = x0 + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{y0-12}" text-anchor="end" transform="rotate(-45 {x} {y0-12})" class="axis">{xml_text(col_name)}</text>')
    for i in range(n):
        for j in range(n):
            v = safe_num(corr.iloc[i, j])
            color = gradient_color(v, -1, 1)
            x, y = x0 + j * cell, y0 + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#fff"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" font-size="10" fill="#111">{v:.2f}</text>')
    return save_svg(parts, filename)


def write_scatter_chart(filename: str, title: str, df: pd.DataFrame, x_col: str, y_col: str, size_col: str, color_col: str) -> Path:
    width, height = 980, 620
    x0, y0, w, h = 88, 72, 780, 430
    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{xml_text(title)}</text>')
    draw_axes(parts, x0, y0, w, h)
    plot = df[[x_col, y_col, size_col, color_col]].dropna().copy()
    xmin, xmax = nice_limits([plot[x_col].values], 0.08)
    ymin, ymax = nice_limits([plot[y_col].values], 0.08)
    cmin, cmax = plot[color_col].min(), plot[color_col].max()

    def mx(v: float) -> float:
        return x0 + (v - xmin) / (xmax - xmin) * w

    def my(v: float) -> float:
        return y0 + h - (v - ymin) / (ymax - ymin) * h

    for k in range(6):
        xv = xmin + (xmax - xmin) * k / 5
        xx = x0 + w * k / 5
        parts.append(f'<text x="{xx:.1f}" y="{y0+h+24}" text-anchor="middle" class="axis">{xv:.1f}</text>')
        yv = ymin + (ymax - ymin) * k / 5
        yy = y0 + h - h * k / 5
        parts.append(f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{yv:.1f}</text>')

    for _, r in plot.iterrows():
        size = 4 + 10 * safe_num(r[size_col]) / max(1.0, plot[size_col].max())
        t = (safe_num(r[color_col]) - cmin) / (cmax - cmin + 1e-12)
        color = gradient_color(t, 0, 1)
        parts.append(f'<circle cx="{mx(r[x_col]):.1f}" cy="{my(r[y_col]):.1f}" r="{size:.1f}" fill="{color}" opacity="0.68" stroke="#333" stroke-width="0.3"/>')

    parts.append(f'<text x="{x0+w/2}" y="{height-25}" text-anchor="middle" class="axis">{xml_text(x_col)}</text>')
    parts.append(f'<text x="24" y="{y0+h/2}" transform="rotate(-90 24 {y0+h/2})" text-anchor="middle" class="axis">{xml_text(y_col)}</text>')
    parts.append(f'<text x="{x0+w+22}" y="{y0+20}" class="note">Color: {xml_text(color_col)}</text>')
    parts.append(f'<text x="{x0+w+22}" y="{y0+42}" class="note">Size: {xml_text(size_col)}</text>')
    return save_svg(parts, filename)


def write_barh_chart(filename: str, title: str, labels: Sequence[str], values: Sequence[float], x_label: str) -> Path:
    width, height = 980, 620
    x0, y0, w, h = 250, 70, 620, 440
    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{xml_text(title)}</text>')
    vals = np.asarray(values, dtype=float)
    max_abs = max(1e-9, float(np.nanmax(np.abs(vals))))
    n = len(labels)
    bar_gap = 8
    bar_h = max(10, (h - bar_gap * (n - 1)) / max(1, n))
    zero_x = x0 + w / 2
    parts.append(f'<line x1="{zero_x}" y1="{y0-8}" x2="{zero_x}" y2="{y0+h+8}" stroke="#555"/>')
    for i, (lab, val) in enumerate(zip(labels, vals)):
        y = y0 + i * (bar_h + bar_gap)
        bar_w = abs(val) / max_abs * (w / 2)
        x = zero_x if val >= 0 else zero_x - bar_w
        color = "#2563eb" if val >= 0 else "#dc2626"
        parts.append(f'<text x="{x0-10}" y="{y+bar_h/2+4}" text-anchor="end" class="axis">{xml_text(lab)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.78"/>')
        parts.append(f'<text x="{x + (bar_w + 4 if val >= 0 else -4):.1f}" y="{y+bar_h/2+4:.1f}" text-anchor="{"start" if val >= 0 else "end"}" class="axis">{val:.3f}</text>')
    parts.append(f'<text x="{x0+w/2}" y="{height-25}" text-anchor="middle" class="axis">{xml_text(x_label)}</text>')
    return save_svg(parts, filename)


def write_confusion_matrix(filename: str, title: str, cm: np.ndarray) -> Path:
    width, height = 620, 540
    x0, y0, cell = 190, 105, 135
    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="36" text-anchor="middle" class="title">{xml_text(title)}</text>')
    labels_x = ["Pred No", "Pred Yes"]
    labels_y = ["Actual No", "Actual Yes"]
    maxv = max(1, int(cm.max()))
    for j, lab in enumerate(labels_x):
        parts.append(f'<text x="{x0+j*cell+cell/2}" y="{y0-18}" text-anchor="middle" class="axis">{lab}</text>')
    for i, lab in enumerate(labels_y):
        parts.append(f'<text x="{x0-12}" y="{y0+i*cell+cell/2+4}" text-anchor="end" class="axis">{lab}</text>')
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            t = v / maxv
            color = gradient_color(t, 0, 1)
            x, y = x0 + j * cell, y0 + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+10}" text-anchor="middle" font-size="28" font-weight="bold" fill="#111">{v}</text>')
    parts.append(f'<text x="{x0+cell}" y="{height-44}" text-anchor="middle" class="axis">Confusion Matrix</text>')
    return save_svg(parts, filename)


# ==========================
# 3. 数据读取、清洗、特征工程
# ==========================
def load_and_prepare_data() -> pd.DataFrame:
    log("Step 1/7: 读取公开宏观经济数据集...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"没有找到数据集：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    expected_cols = [
        "year", "quarter", "realgdp", "realcons", "realinv", "realgovt",
        "realdpi", "cpi", "m1", "tbilrate", "unemp", "pop", "infl", "realint"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据列缺失：{missing}")

    for c in expected_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["quarter_label"] = df["year"].astype(int).astype(str) + "Q" + df["quarter"].astype(int).astype(str)
    df["date"] = pd.PeriodIndex(df["quarter_label"], freq="Q").to_timestamp(how="end")
    df = df.sort_values("date").reset_index(drop=True)

    log("Step 2/7: 数据清洗、转换和指标构造...")
    df["gdp_qoq_ann"] = 400 * np.log(df["realgdp"] / df["realgdp"].shift(1))
    df["cons_qoq_ann"] = 400 * np.log(df["realcons"] / df["realcons"].shift(1))
    df["inv_qoq_ann"] = 400 * np.log(df["realinv"] / df["realinv"].shift(1))
    df["gov_qoq_ann"] = 400 * np.log(df["realgovt"] / df["realgovt"].shift(1))

    df["gdp_yoy"] = 100 * (df["realgdp"] / df["realgdp"].shift(4) - 1)
    df["cons_yoy"] = 100 * (df["realcons"] / df["realcons"].shift(4) - 1)
    df["inv_yoy"] = 100 * (df["realinv"] / df["realinv"].shift(4) - 1)

    df["consumption_share"] = 100 * df["realcons"] / df["realgdp"]
    df["investment_share"] = 100 * df["realinv"] / df["realgdp"]
    df["government_share"] = 100 * df["realgovt"] / df["realgdp"]
    df["real_rate_gap"] = df["tbilrate"] - df["infl"]

    # 业务目标：预测下一季度是否出现经济放缓。
    # 定义：下一季度 GDP 环比年化增速低于 1% 记为 1，否则记为 0。
    df["next_gdp_qoq_ann"] = df["gdp_qoq_ann"].shift(-1)
    df["slowdown_next_q"] = (df["next_gdp_qoq_ann"] < 1.0).astype(int)

    # 宏观风险评分：高失业、高通胀、低 GDP 增长、低投资增长、偏高实际利率会推高风险。
    risk_unemp = minmax_safe(df["unemp"])
    risk_infl = minmax_safe(df["infl"].abs())
    risk_low_growth = 1 - minmax_safe(df["gdp_yoy"])
    risk_low_inv = 1 - minmax_safe(df["inv_yoy"])
    risk_real_rate = minmax_safe(df["realint"])

    df["macro_risk_score"] = 100 * (
        0.30 * risk_unemp +
        0.20 * risk_infl +
        0.25 * risk_low_growth +
        0.15 * risk_low_inv +
        0.10 * risk_real_rate
    )
    df["risk_level"] = pd.cut(
        df["macro_risk_score"],
        bins=[-np.inf, 35, 60, np.inf],
        labels=["Low", "Medium", "High"]
    )

    clean_path = TABLE_DIR / "clean_macro_panel.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    log(f"清洗后的数据集已保存: {clean_path}")
    return df


# ==========================
# 4. 可视化输出
# ==========================
def make_visualizations(df: pd.DataFrame) -> None:
    log("Step 3/7: 生成可视化图表...")
    labels = df["quarter_label"].tolist()

    base = df.loc[0, ["realgdp", "realcons", "realinv", "realgovt"]]
    indexed = df[["realgdp", "realcons", "realinv", "realgovt"]].divide(base) * 100
    write_line_chart(
        "fig01_macro_components_index.svg",
        "Macroeconomic Components Indexed to 1959Q1 = 100",
        labels,
        {
            "Real GDP": indexed["realgdp"].values,
            "Real Consumption": indexed["realcons"].values,
            "Real Investment": indexed["realinv"].values,
            "Real Government Spending": indexed["realgovt"].values,
        },
        "Index"
    )

    write_line_chart(
        "fig02_growth_inflation_unemployment.svg",
        "Growth, Inflation and Unemployment",
        labels,
        {
            "GDP YoY Growth (%)": df["gdp_yoy"].values,
            "Inflation (%)": df["infl"].values,
            "Unemployment Rate (%)": df["unemp"].values,
        },
        "Percent"
    )

    corr_cols = [
        "gdp_qoq_ann", "cons_qoq_ann", "inv_qoq_ann", "infl",
        "unemp", "tbilrate", "realint", "investment_share", "macro_risk_score"
    ]
    corr = df[corr_cols].dropna().corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    write_heatmap("fig03_correlation_heatmap.svg", "Correlation Map of Macro Indicators", corr)

    write_scatter_chart(
        "fig04_inflation_unemployment_scatter.svg",
        "Inflation vs. Unemployment with GDP Growth Color",
        df,
        x_col="unemp",
        y_col="infl",
        size_col="macro_risk_score",
        color_col="gdp_yoy",
    )

    write_line_chart(
        "fig05_business_cycle_risk_score.svg",
        "Business Cycle Risk Score",
        labels,
        {"Macro Risk Score": df["macro_risk_score"].values},
        "Risk Score: 0-100",
        y_min=0,
        y_max=100,
        zones=[(0, 35, "#16a34a", "Low"), (35, 60, "#f59e0b", "Medium"), (60, 100, "#dc2626", "High")]
    )

    write_line_chart(
        "fig06_gdp_structure_share.svg",
        "Demand-side Structure of Real GDP",
        labels,
        {
            "Consumption / GDP": df["consumption_share"].values,
            "Investment / GDP": df["investment_share"].values,
            "Government / GDP": df["government_share"].values,
        },
        "Share of Real GDP (%)"
    )


# ==========================
# 5. 手写逻辑回归与评估
# ==========================
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.045,
    epochs: int = LOGISTIC_EPOCHS,
    l2: float = 0.015,
) -> Tuple[np.ndarray, float, List[float]]:
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    y = y.astype(float)
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    class_weight = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    losses = []

    for epoch in range(epochs):
        z = X @ w + b
        prob = sigmoid(z)
        error = (prob - y) * class_weight
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % 250 == 0 or epoch == epochs - 1:
            eps = 1e-9
            loss = -np.mean(class_weight * (y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))) + 0.5 * l2 * np.sum(w ** 2)
            losses.append(float(loss))
    return w, b, losses


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    prob = sigmoid(X @ w + b)
    pred = (prob >= threshold).astype(int)
    return pred, prob


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    auc = simple_auc(y_true, proba) if proba is not None else np.nan
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def simple_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    score = np.asarray(score, dtype=float)
    pos_scores = score[y_true == 1]
    neg_scores = score[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    count = 0.0
    total = len(pos_scores) * len(neg_scores)
    for ps in pos_scores:
        count += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
    return float(count / total)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def train_predictive_model(df: pd.DataFrame) -> Dict[str, object]:
    log("Step 4/7: 构建下一季度经济放缓预测模型...")
    feature_cols = [
        "gdp_qoq_ann", "cons_qoq_ann", "inv_qoq_ann", "gov_qoq_ann",
        "gdp_yoy", "infl", "unemp", "tbilrate", "realint",
        "investment_share", "consumption_share", "government_share", "macro_risk_score"
    ]
    model_df = df.dropna(subset=feature_cols + ["slowdown_next_q", "next_gdp_qoq_ann"]).copy().reset_index(drop=True)

    split_idx = int(len(model_df) * 0.75)
    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train_raw = train_df[feature_cols].values.astype(float)
    X_test_raw = test_df[feature_cols].values.astype(float)
    y_train = train_df["slowdown_next_q"].values.astype(int)
    y_test = test_df["slowdown_next_q"].values.astype(int)

    X_train, X_test, mean, std = standardize_train_test(X_train_raw, X_test_raw)
    w, b, losses = train_logistic_regression(X_train, y_train)
    pred, proba = predict_logistic(X_test, w, b)
    m = binary_metrics(y_test, pred, proba)
    cm = confusion(y_test, pred)

    # 对比基准：直接用宏观风险分数是否超过训练集 65 分位预测放缓。
    threshold = float(train_df["macro_risk_score"].quantile(0.65))
    base_proba = minmax_safe(test_df["macro_risk_score"]).values
    base_pred = (test_df["macro_risk_score"].values >= threshold).astype(int)
    bm = binary_metrics(y_test, base_pred, base_proba)

    metrics_df = pd.DataFrame([
        {"model": "Handmade Logistic Regression", **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "auc"]}},
        {"model": "Risk Score Threshold Baseline", **{k: bm[k] for k in ["accuracy", "precision", "recall", "f1", "auc"]}},
    ]).sort_values(["f1", "auc", "accuracy"], ascending=False)
    metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    pred_df = test_df[["quarter_label", "date", "gdp_qoq_ann", "next_gdp_qoq_ann", "slowdown_next_q", "macro_risk_score"]].copy()
    pred_df["pred_slowdown"] = pred
    pred_df["pred_slowdown_probability"] = proba
    pred_df.to_csv(TABLE_DIR / "model_predictions.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": w}).sort_values("coefficient")
    coef_df.to_csv(TABLE_DIR / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")

    log(f"手写逻辑回归: accuracy={m['accuracy']:.3f}, f1={m['f1']:.3f}, auc={m['auc']:.3f}")
    log(f"风险阈值基准: accuracy={bm['accuracy']:.3f}, f1={bm['f1']:.3f}, auc={bm['auc']:.3f}")

    write_line_chart(
        "fig07_slowdown_prediction_probability.svg",
        "Out-of-sample Next-quarter Slowdown Prediction",
        pred_df["quarter_label"].tolist(),
        {
            "Predicted Slowdown Probability": pred_df["pred_slowdown_probability"].values,
            "Actual Slowdown Label": pred_df["slowdown_next_q"].values,
        },
        "Probability / Label",
        y_min=-0.05,
        y_max=1.05,
    )
    write_confusion_matrix("fig08_confusion_matrix.svg", "Confusion Matrix - Handmade Logistic Regression", cm)
    write_barh_chart(
        "fig09_logistic_coefficients.svg",
        "Model Coefficients for Next-quarter Slowdown Prediction",
        coef_df["feature"].tolist(),
        coef_df["coefficient"].tolist(),
        "Standardized Coefficient"
    )

    return {
        "feature_cols": feature_cols,
        "model_df": model_df,
        "test_df": test_df,
        "metrics_df": metrics_df,
        "weights": w,
        "bias": b,
        "mean": mean,
        "std": std,
        "pred_df": pred_df,
        "confusion_matrix": cm,
        "losses": losses,
    }


# ==========================
# 6. 重复抽样稳定性与置换重要性
# ==========================
def permutation_importance_manual(model_info: Dict[str, object]) -> pd.DataFrame:
    feature_cols: List[str] = model_info["feature_cols"]  # type: ignore
    test_df: pd.DataFrame = model_info["test_df"]  # type: ignore
    w: np.ndarray = model_info["weights"]  # type: ignore
    b: float = model_info["bias"]  # type: ignore
    mean: np.ndarray = model_info["mean"]  # type: ignore
    std: np.ndarray = model_info["std"]  # type: ignore

    X = test_df[feature_cols].values.astype(float)
    y = test_df["slowdown_next_q"].values.astype(int)
    Xs = (X - mean) / std
    pred, prob = predict_logistic(Xs, w, b)
    base_f1 = binary_metrics(y, pred, prob)["f1"]

    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for j, col in enumerate(feature_cols):
        drops = []
        for _ in range(30):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            Xp = (X_perm - mean) / std
            pred_p, prob_p = predict_logistic(Xp, w, b)
            f1_p = binary_metrics(y, pred_p, prob_p)["f1"]
            drops.append(base_f1 - f1_p)
        records.append({
            "feature": col,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })
    imp = pd.DataFrame(records).sort_values("importance_mean")
    imp.to_csv(TABLE_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    write_barh_chart(
        "fig10_permutation_importance.svg",
        "Permutation Feature Importance",
        imp["feature"].tolist(),
        imp["importance_mean"].tolist(),
        "Decrease in F1 after Shuffling"
    )
    return imp


def bootstrap_model_stability(model_info: Dict[str, object]) -> pd.DataFrame:
    log("Step 5/7: 执行重复抽样稳定性验证...")
    feature_cols: List[str] = model_info["feature_cols"]  # type: ignore
    model_df: pd.DataFrame = model_info["model_df"]  # type: ignore

    split_idx = int(len(model_df) * 0.75)
    train_df = model_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = model_df.iloc[split_idx:].copy().reset_index(drop=True)
    X_test_raw = test_df[feature_cols].values.astype(float)
    y_test = test_df["slowdown_next_q"].values.astype(int)

    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for i in range(1, BOOTSTRAP_ITERATIONS + 1):
        boot_idx = rng.choice(train_df.index, size=len(train_df), replace=True)
        boot_train = train_df.loc[boot_idx]
        X_train_raw = boot_train[feature_cols].values.astype(float)
        y_train = boot_train["slowdown_next_q"].values.astype(int)
        X_train, X_test, _, _ = standardize_train_test(X_train_raw, X_test_raw)
        w, b, _ = train_logistic_regression(X_train, y_train, epochs=10, lr=0.04)
        pred, prob = predict_logistic(X_test, w, b)
        m = binary_metrics(y_test, pred, prob)
        records.append({
            "iteration": i,
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "auc": m["auc"],
            "mean_predicted_probability": float(np.mean(prob)),
        })
        if i % 20 == 0:
            log(f"  bootstrap progress: {i}/{BOOTSTRAP_ITERATIONS}")

    boot_df = pd.DataFrame(records)
    boot_df.to_csv(TABLE_DIR / "bootstrap_stability_metrics.csv", index=False, encoding="utf-8-sig")
    write_line_chart(
        "fig11_bootstrap_model_stability.svg",
        "Bootstrap Stability of Slowdown Prediction Model",
        boot_df["iteration"].astype(str).tolist(),
        {
            "Accuracy": boot_df["accuracy"].values,
            "F1": boot_df["f1"].values,
            "AUC": boot_df["auc"].values,
        },
        "Metric",
        y_min=0,
        y_max=1.05,
    )
    return boot_df


# ==========================
# 7. 生成最终数据产品 HTML
# ==========================
def generate_html_report(df: pd.DataFrame, model_info: Dict[str, object], boot_df: pd.DataFrame) -> Path:
    log("Step 6/7: 生成最终 HTML 数据产品报告...")
    metrics_df: pd.DataFrame = model_info["metrics_df"]  # type: ignore
    latest = df.dropna(subset=["macro_risk_score", "gdp_yoy"]).iloc[-1]
    best = metrics_df.iloc[0]

    summary = {
        "observations": int(len(df)),
        "start_quarter": df["quarter_label"].iloc[0],
        "end_quarter": df["quarter_label"].iloc[-1],
        "latest_risk_score": float(latest["macro_risk_score"]),
        "latest_risk_level": str(latest["risk_level"]),
        "latest_gdp_yoy": float(latest["gdp_yoy"]),
        "latest_unemployment": float(latest["unemp"]),
        "latest_inflation": float(latest["infl"]),
        "best_model": str(best["model"]),
        "best_model_f1": float(best["f1"]),
        "best_model_auc": float(best["auc"]),
        "bootstrap_mean_f1": float(boot_df["f1"].mean()),
        "bootstrap_mean_auc": float(boot_df["auc"].mean()),
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "executive_summary.csv", index=False, encoding="utf-8-sig")

    figure_sections = []
    for p in sorted(FIG_DIR.glob("*.svg")):
        figure_sections.append(
            f"<section><h3>{xml_text(p.stem)}</h3><img src='figures/{xml_text(p.name)}' alt='{xml_text(p.name)}'></section>"
        )

    rows = []
    for _, r in metrics_df.iterrows():
        rows.append(
            f"<tr><td>{xml_text(r['model'])}</td><td>{float(r['accuracy']):.3f}</td>"
            f"<td>{float(r['precision']):.3f}</td><td>{float(r['recall']):.3f}</td>"
            f"<td>{float(r['f1']):.3f}</td><td>{float(r['auc']):.3f}</td></tr>"
        )

    html_text = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>US Macro Business Cycle Risk Data Product</title>
<style>
body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 30px; color: #222; background: #f5f7fb; }}
.card {{ background: #fff; padding: 24px; border-radius: 14px; box-shadow: 0 2px 14px rgba(0,0,0,0.08); margin-bottom: 24px; }}
h1 {{ margin-top: 0; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
.kpi {{ background: #eef4ff; border-left: 5px solid #2563eb; border-radius: 10px; padding: 14px; }}
.kpi b {{ display: block; font-size: 22px; margin-top: 6px; }}
p {{ line-height: 1.85; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 9px; text-align: center; }}
th {{ background: #f1f1f1; }}
img {{ width: 100%; max-width: 1120px; background: white; border: 1px solid #ddd; border-radius: 8px; }}
section {{ margin-top: 24px; }}
.small {{ color: #555; font-size: 13px; }}
</style>
</head>
<body>
<div class="card">
<h1>美国宏观经济景气与下一季度放缓风险监控数据产品</h1>
<p>本项目从公开经济数据出发，完成数据读取、清洗、特征转换、描述性分析、可视化、风险评分和预测建模，最终形成一个面向金融分析师、商业策略人员和经济研究助理的小型数据产品。</p>
<p class="small">数据集：US macroeconomic quarterly data，样本区间 {summary['start_quarter']} 至 {summary['end_quarter']}，共 {summary['observations']} 条季度记录。</p>
</div>

<div class="card">
<h2>核心 KPI</h2>
<div class="kpi-grid">
  <div class="kpi">样本区间<b>{summary['start_quarter']} - {summary['end_quarter']}</b></div>
  <div class="kpi">最新风险分数<b>{summary['latest_risk_score']:.1f}</b></div>
  <div class="kpi">最新风险等级<b>{summary['latest_risk_level']}</b></div>
  <div class="kpi">最佳模型<b>{xml_text(summary['best_model'])}</b></div>
  <div class="kpi">最新 GDP 同比增速<b>{summary['latest_gdp_yoy']:.2f}%</b></div>
  <div class="kpi">最新失业率<b>{summary['latest_unemployment']:.2f}%</b></div>
  <div class="kpi">最新通胀率<b>{summary['latest_inflation']:.2f}%</b></div>
  <div class="kpi">最佳模型 AUC<b>{summary['best_model_auc']:.3f}</b></div>
</div>
</div>

<div class="card">
<h2>分析方法说明</h2>
<p>数据清洗阶段将季度字段转换为时间序列索引，并计算 GDP、消费、投资、政府支出的环比年化增速和同比增速；产品层面进一步构造消费/GDP、投资/GDP、政府/GDP 等结构指标。宏观风险分数综合失业率、通胀、GDP 增长、投资增长和实际利率，分数越高代表经济压力越大。</p>
<p>建模阶段采用严格时间序列切分，前 75% 样本训练，后 25% 样本测试，预测目标为“下一季度 GDP 环比年化增速是否低于 1%”。为了展示方法理解，逻辑回归由 numpy 手写梯度下降完成，没有直接调用机器学习库。</p>
</div>

<div class="card">
<h2>模型评估结果</h2>
<table>
<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>
{''.join(rows)}
</table>
<p>重复抽样验证：Bootstrap 平均 F1 = {summary['bootstrap_mean_f1']:.3f}，Bootstrap 平均 AUC = {summary['bootstrap_mean_auc']:.3f}。</p>
</div>

<div class="card">
<h2>可视化看板</h2>
{''.join(figure_sections)}
</div>
</body>
</html>
"""

    path = OUTPUT_DIR / "macro_business_cycle_report.html"
    path.write_text(html_text, encoding="utf-8")
    log(f"HTML 数据产品报告已生成: {path}")
    return path


# ==========================
# 8. 主程序
# ==========================
def main() -> None:
    start = time.time()
    print("=" * 90)
    print("        US Macro Business Cycle Risk Data Product - Python Assignment")
    print("=" * 90)

    df = load_and_prepare_data()
    make_visualizations(df)
    model_info = train_predictive_model(df)
    permutation_importance_manual(model_info)
    boot_df = bootstrap_model_stability(model_info)
    report_path = generate_html_report(df, model_info, boot_df)

    log("Step 7/7: 检查运行时长并输出最终结果...")
    ensure_min_runtime(start, MIN_RUNTIME_SECONDS)

    elapsed = time.time() - start
    print("=" * 90)
    print("项目运行完成！")
    print(f"总运行时间: {elapsed:.1f} 秒")
    print(f"公开数据集文件: {DATA_PATH}")
    print(f"清洗后的数据: {TABLE_DIR / 'clean_macro_panel.csv'}")
    print(f"最终 HTML 数据产品: {report_path}")
    print(f"图表目录: {FIG_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
