import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from src.models.trainer import FEATURE_COLUMNS, TARGET_COLUMN


CHART_STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#2e2e3a",
    "axes.labelcolor":   "#e0e0e0",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#a0a0b0",
    "ytick.color":       "#a0a0b0",
    "grid.color":        "#2e2e3a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        "#e0e0e0",
    "font.family":       "DejaVu Sans",
}

ACCENT      = "#7c6af7"
ACCENT_2    = "#f7a76c"
ACCENT_3    = "#6cf7c5"
ACTUAL_COL  = "#a0a0c0"


def _apply_style():
    plt.rcParams.update(CHART_STYLE)


def _save(fig: plt.Figure, filename: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[charts] ✓ Saved: {path}")
    return path


def chart_sales_history(
    features_df: pd.DataFrame,
    output_dir: str = "outputs/charts"
) -> str:
    _apply_style()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        features_df["week_start"],
        features_df["total_sales"],
        color=ACTUAL_COL, linewidth=1.2, alpha=0.7, label="Weekly Sales"
    )

    rolling = features_df["total_sales"].rolling(8, center=True).mean()
    ax.plot(
        features_df["week_start"],
        rolling,
        color=ACCENT, linewidth=2.5, label="8-Week Trend"
    )

    ax.set_title("Weekly Sales History  (2014 – 2017)", fontsize=16, pad=16)
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Sales ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=35)
    ax.legend(framealpha=0.15)
    ax.grid(True)
    fig.tight_layout()

    return _save(fig, "01_sales_history.png", output_dir)


def chart_forecast(
    features_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_dir: str = "outputs/charts"
) -> str:
    _apply_style()

    fig, ax = plt.subplots(figsize=(14, 5))

    history_tail = features_df.tail(26)
    ax.plot(
        history_tail["week_start"],
        history_tail["total_sales"],
        color=ACTUAL_COL, linewidth=1.4,
        label="Historical Sales (last 26 weeks)"
    )

    ax.plot(
        forecast_df["week_start"],
        forecast_df["predicted_sales"],
        color=ACCENT_2, linewidth=2.5,
        linestyle="--", label="Forecast"
    )

    std_val = features_df["total_sales"].tail(26).std()
    ax.fill_between(
        forecast_df["week_start"],
        forecast_df["predicted_sales"] - std_val * 0.6,
        forecast_df["predicted_sales"] + std_val * 0.6,
        color=ACCENT_2, alpha=0.12, label="Confidence Band"
    )

    ax.axvline(
        x=forecast_df["week_start"].min(),
        color="#ff6b6b", linewidth=1.5,
        linestyle=":", label="Forecast Start"
    )

    total_rev = forecast_df["predicted_sales"].sum()
    ax.set_title(
        f"12-Week Sales Forecast  |  Projected Revenue: ${total_rev:,.0f}",
        fontsize=15, pad=16
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Sales ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=35)
    ax.legend(framealpha=0.15)
    ax.grid(True)
    fig.tight_layout()

    return _save(fig, "02_forecast.png", output_dir)


def chart_seasonality(
    features_df: pd.DataFrame,
    output_dir: str = "outputs/charts"
) -> str:
    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    month_avg = (
        features_df.groupby("month")["total_sales"]
        .mean()
        .reset_index()
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_avg["month_name"] = month_avg["month"].apply(
        lambda m: month_names[m - 1]
    )

    colors_m = [
        ACCENT if v == month_avg["total_sales"].max()
        else ACCENT_2 if v == month_avg["total_sales"].min()
        else "#3a3a5c"
        for v in month_avg["total_sales"]
    ]

    axes[0].bar(
        month_avg["month_name"],
        month_avg["total_sales"],
        color=colors_m, edgecolor="#0f1117", linewidth=0.5
    )
    axes[0].set_title("Avg Weekly Sales by Month", fontsize=13)
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Avg Sales ($)")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y")

    quarter_avg = (
        features_df.groupby("quarter")["total_sales"]
        .mean()
        .reset_index()
    )
    q_labels  = [f"Q{q}" for q in quarter_avg["quarter"]]
    colors_q  = [ACCENT, "#3a3a5c", "#3a3a5c", ACCENT_3]

    axes[1].bar(
        q_labels,
        quarter_avg["total_sales"],
        color=colors_q, edgecolor="#0f1117", linewidth=0.5,
        width=0.5
    )
    axes[1].set_title("Avg Weekly Sales by Quarter", fontsize=13)
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel("Avg Sales ($)")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[1].grid(True, axis="y")

    fig.suptitle("Seasonality Breakdown", fontsize=16, y=1.02)
    fig.tight_layout()

    return _save(fig, "03_seasonality.png", output_dir)


def chart_model_performance(
    features_df: pd.DataFrame,
    model,
    split_date: pd.Timestamp,
    metrics: dict,
    output_dir: str = "outputs/charts"
) -> str:
    _apply_style()

    test_df = features_df[
        features_df["week_start"] >= split_date
    ].copy()

    X_test      = test_df[FEATURE_COLUMNS]
    actual      = test_df[TARGET_COLUMN].values
    predicted   = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        test_df["week_start"], actual,
        color=ACTUAL_COL, linewidth=1.5, label="Actual"
    )
    axes[0].plot(
        test_df["week_start"], predicted,
        color=ACCENT_2, linewidth=2,
        linestyle="--", label="Predicted"
    )
    axes[0].set_title("Actual vs Predicted  (Test Period)", fontsize=13)
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Sales ($)")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=35)
    axes[0].legend(framealpha=0.15)
    axes[0].grid(True)

    metric_names  = ["MAE", "RMSE", "MAPE (%)", "R²"]
    metric_values = [
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        metrics["R2"]
    ]
    bar_colors = [
        ACCENT_3 if v < 5000 or k == "R²" else ACCENT_2
        for k, v in zip(metric_names, metric_values)
    ]

    bars = axes[1].bar(
        metric_names, metric_values,
        color=bar_colors, edgecolor="#0f1117",
        linewidth=0.5, width=0.45
    )
    for bar, val in zip(bars, metric_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(metric_values) * 0.01,
            f"{val:,.2f}",
            ha="center", va="bottom",
            fontsize=10, color="#e0e0e0"
        )
    axes[1].set_title("Model Evaluation Metrics", fontsize=13)
    axes[1].set_ylabel("Value")
    axes[1].grid(True, axis="y")

    fig.suptitle(
        f"Model Performance  |  R²: {metrics['R2']}  |  MAPE: {metrics['MAPE']:.1f}%",
        fontsize=15, y=1.02
    )
    fig.tight_layout()

    return _save(fig, "04_model_performance.png", output_dir)