import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

TARGET_KEYWORDS = ["revenue", "sales", "close", "price", "income", "profit", "amount", "total"]


def load_file(file, filename):
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def find_date_col(df):
    for c in df.columns:
        if any(k in str(c).lower() for k in ["date", "month", "period", "time", "day", "year"]):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
                return c
    return None


def pick_target(num_cols):
    for kw in TARGET_KEYWORDS:
        for c in num_cols:
            if kw in str(c).lower():
                return c
    return num_cols[0]


def find_col(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw in str(c).lower():
                return c
    return None


def analyze(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = find_date_col(df)
    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return {"error": "No numeric columns found in this file to analyze."}

    target = pick_target(num_cols)
    if date_col:
        labels = df[date_col].dt.strftime("%Y-%m-%d").tolist()
    else:
        labels = [f"Period {i+1}" for i in range(len(df))]

    findings = []
    structured_findings = []

    # 1. Overview
    missing = int(df.isna().sum().sum())
    overview_text = (f"Dataset contains {len(df):,} records and {len(df.columns)} columns. "
                     f"Detected {missing} missing cell(s). Primary key metric selected: '{target}'.")
    findings.append(overview_text)
    structured_findings.append({
        "category": "Data Health",
        "title": "Dataset Scope & Metric Target",
        "text": overview_text,
        "type": "info",
        "icon": "database"
    })

    # 2. Target metric summary
    y = df[target].ffill().bfill().values.astype(float)
    y_mean = float(y.mean()) if len(y) else 0.0
    y_min = float(y.min()) if len(y) else 0.0
    y_max = float(y.max()) if len(y) else 0.0
    y_sum = float(y.sum()) if len(y) else 0.0
    y_latest = float(y[-1]) if len(y) else 0.0

    summary_text = (f"'{target}' baseline: Average of {y_mean:,.2f}, Range spanning from "
                    f"{y_min:,.2f} to peak {y_max:,.2f}, accumulating to {y_sum:,.2f} in total.")
    findings.append(summary_text)
    structured_findings.append({
        "category": "Performance",
        "title": f"'{target}' Metric Baseline",
        "text": summary_text,
        "type": "primary",
        "icon": "activity"
    })

    # 3. Trend (Linear Regression)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    pct = float((y[-1] - y[0]) / abs(y[0]) * 100) if len(y) > 0 and y[0] != 0 else 0.0
    direction = "upward" if slope > 0 else "downward"

    trend_text = (f"Trend trajectory is {direction.upper()}: '{target}' registered a {pct:+.1f}% net "
                  f"shift from inception to latest record (approx. {slope:+,.2f} velocity per period).")
    findings.append(trend_text)
    structured_findings.append({
        "category": "Trend Velocity",
        "title": f"Growth Vector ({direction.capitalize()})",
        "text": trend_text,
        "type": "success" if slope >= 0 else "warning",
        "icon": "trending-up" if slope >= 0 else "trending-down"
    })

    # 4. Volatility
    cv = float(y.std() / abs(y_mean) * 100) if y_mean != 0 else 0.0
    vol_level = "High" if cv > 30 else "Moderate" if cv > 10 else "Low"
    vol_text = f"Volatility is assessed as {vol_level.upper()} (Standard deviation represents {cv:.1f}% of the average metric volume)."
    findings.append(vol_text)
    structured_findings.append({
        "category": "Risk Index",
        "title": f"{vol_level} Volatility Exposure",
        "text": vol_text,
        "type": "warning" if vol_level == "High" else "info",
        "icon": "bar-chart-2"
    })

    # 5. Profit margin
    margin = None
    rev = find_col(num_cols, ["revenue", "sales"])
    prof = find_col(num_cols, ["profit", "net income", "margin"])
    if rev and prof and rev != prof and df[rev].sum() != 0:
        margin = float(df[prof].sum() / df[rev].sum() * 100)
        margin_text = f"Aggregated profitability index ({prof} / {rev}) stands at {margin:.1f}%."
        findings.append(margin_text)
        structured_findings.append({
            "category": "Profitability",
            "title": "Consolidated Margin Efficiency",
            "text": margin_text,
            "type": "success" if margin > 15 else "info",
            "icon": "pie-chart"
        })

    # 6. Correlations
    correlations_list = []
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1 = num_cols[i]
                c2 = num_cols[j]
                val = float(corr_matrix.loc[c1, c2])
                if not np.isnan(val):
                    strength = "Strong Positive" if val > 0.7 else "Moderate Positive" if val > 0.3 else "Strong Negative" if val < -0.7 else "Moderate Negative" if val < -0.3 else "Neutral"
                    correlations_list.append({
                        "var1": c1,
                        "var2": c2,
                        "score": round(val, 3),
                        "abs_score": round(abs(val), 3),
                        "strength": strength
                    })
        correlations_list.sort(key=lambda x: x["abs_score"], reverse=True)
        if correlations_list:
            top_corr = correlations_list[0]
            corr_text = f"Dominant relationship identified: '{top_corr['var1']}' and '{top_corr['var2']}' (correlation coefficient of {top_corr['score']:+.2f})."
            findings.append(corr_text)
            structured_findings.append({
                "category": "Cross Correlation",
                "title": "Leading Variable Synergy",
                "text": corr_text,
                "type": "primary",
                "icon": "git-merge"
            })

    # 7. Anomalies (Isolation Forest)
    anomalies = []
    anomalies_details = []
    if len(df) >= 10:
        data = df[num_cols].ffill().bfill().fillna(0)
        flags = IsolationForest(contamination=0.05, random_state=42).fit_predict(data)
        idx = np.where(flags == -1)[0]
        anomalies = [str(labels[i]) for i in idx]
        for i in idx:
            anomalies_details.append({
                "index": int(i) + 1,
                "label": str(labels[i]),
                "value": float(df[target].iloc[i]),
                "note": "Deviates significantly from standard cluster distribution"
            })
        if anomalies:
            anom_text = (f"{len(anomalies)} statistical outlier(s) detected at periods: "
                         f"{', '.join(anomalies[:4])}{'...' if len(anomalies) > 4 else ''}. Audit recommended.")
            findings.append(anom_text)
            structured_findings.append({
                "category": "Anomaly Watch",
                "title": f"{len(anomalies)} Potential Outliers Flagged",
                "text": anom_text,
                "type": "danger",
                "icon": "alert-triangle"
            })
        else:
            findings.append("No statistical anomalies or cluster outliers identified in dataset.")
            structured_findings.append({
                "category": "Anomaly Watch",
                "title": "No Significant Anomalies",
                "text": "Dataset records align consistently with baseline variance bounds.",
                "type": "success",
                "icon": "check-circle"
            })

    # 8. Forecast next 3 periods
    future_X = np.arange(len(y), len(y) + 3).reshape(-1, 1)
    raw_forecast = model.predict(future_X).tolist()
    forecast = [round(float(v), 2) for v in raw_forecast]

    forecast_labels = []
    if date_col and len(df) > 1:
        last_dt = df[date_col].iloc[-1]
        freq = pd.infer_freq(df[date_col])
        for step in range(1, 4):
            try:
                if freq:
                    fut_dt = pd.date_range(start=last_dt, periods=step + 1, freq=freq)[-1]
                    forecast_labels.append(fut_dt.strftime("%Y-%m-%d"))
                else:
                    fut_dt = last_dt + pd.DateOffset(months=step)
                    forecast_labels.append(fut_dt.strftime("%Y-%m-%d"))
            except Exception:
                forecast_labels.append(f"Projection +{step}")
    else:
        forecast_labels = [f"Projection +{i+1}" for i in range(3)]

    forecast_items = []
    prev_val = y_latest
    for idx_f, (f_lbl, f_val) in enumerate(zip(forecast_labels, forecast)):
        delta_pct = ((f_val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0.0
        forecast_items.append({
            "step": idx_f + 1,
            "label": f_lbl,
            "value": f_val,
            "delta_pct": round(delta_pct, 1),
            "trend": "up" if f_val >= prev_val else "down"
        })
        prev_val = f_val

    forecast_text = f"Forward projections for next 3 cycles: {', '.join(f'{v:,.2f}' for v in forecast)}."
    findings.append(forecast_text)
    structured_findings.append({
        "category": "Forecasting",
        "title": "Next 3 Cycles Linear Projection",
        "text": forecast_text,
        "type": "primary",
        "icon": "compass"
    })

    # Multi-series data for comparison chart
    series_data = {}
    for col in num_cols:
        col_series = df[col].ffill().bfill().values.astype(float).tolist()
        series_data[col] = [round(v, 2) for v in col_series]

    # Period-over-period growth rates for target
    period_growth = []
    for i in range(len(y)):
        if i == 0:
            period_growth.append(0.0)
        else:
            prev = y[i - 1]
            growth = round(((y[i] - prev) / abs(prev) * 100), 2) if prev != 0 else 0.0
            period_growth.append(growth)

    # Column statistical summaries
    col_stats = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s):
            col_stats.append({
                "name": col,
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "sum": round(float(s.sum()), 2),
                "is_target": (col == target)
            })

    # Table preview
    display_df = df.copy()
    if date_col:
        display_df[date_col] = display_df[date_col].dt.strftime("%Y-%m-%d")

    preview_rows = display_df.head(25).to_dict(orient="records")
    preview_cols = list(display_df.columns)

    return {
        # Backwards compatibility
        "findings": findings,
        "target": target,
        "labels": [str(l) for l in labels],
        "actual": [round(float(v), 2) for v in y],
        "forecast": forecast,
        "preview": df.head(5).to_html(classes="table", index=False),

        # Enhanced dashboard payload
        "structured_findings": structured_findings,
        "forecast_labels": forecast_labels,
        "forecast_items": forecast_items,
        "series_data": series_data,
        "numeric_columns": num_cols,
        "period_growth": period_growth,
        "correlations": correlations_list[:6],
        "anomalies": anomalies,
        "anomalies_details": anomalies_details,
        "column_stats": col_stats,
        "preview_rows": preview_rows,
        "preview_cols": preview_cols,
        "metrics": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "missing_values": missing,
            "target_metric": target,
            "target_mean": y_mean,
            "target_min": y_min,
            "target_max": y_max,
            "target_sum": y_sum,
            "target_latest": y_latest,
            "growth_pct": round(pct, 1),
            "slope": round(slope, 2),
            "direction": direction,
            "volatility_level": vol_level,
            "volatility_cv": round(cv, 1),
            "anomalies_count": len(anomalies),
            "profit_margin": round(margin, 1) if margin is not None else None
        }
    }