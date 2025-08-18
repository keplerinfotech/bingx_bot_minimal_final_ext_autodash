from __future__ import annotations

import os

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


def build_dashboard(csv_path: str, out_html: str = "dashboard.html") -> str:
    df = pd.read_csv(csv_path)

    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for col in (
        "filled",
        "filled_qty",
        "queue_ahead",
        "agg_consumed",
        "price",
        "qty",
        "route_prob",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby("venue")
        .agg(
            filled_rate=("filled", "mean"),
            orders=("order_id", "count"),
            avg_fill_qty=("filled_qty", "mean"),
            avg_queue_ahead=("queue_ahead", "mean"),
            avg_route_prob=("route_prob", "mean"),
        )
        .reset_index()
    )

    if px is None or go is None:
        html = "<html><head><title>Final Fill Quality Report</title></head><body>"
        html += "<h2>Summary by Venue</h2>"
        html += summary.to_html(index=False)
        html += "<h2>Sample Fills (first 200)</h2>"
        html += df.head(200).to_html(index=False)
        html += "</body></html>"
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        return out_html

    figs = []
    fig_filled = px.bar(
        summary,
        x="venue",
        y="filled_rate",
        title="Filled Rate by Venue",
        text="filled_rate",
    )
    fig_filled.update_layout(yaxis_tickformat=".0%")
    figs.append(fig_filled)

    fig_orders = px.bar(
        summary, x="venue", y="orders", title="Orders by Venue", text="orders"
    )
    figs.append(fig_orders)

    fig_prob = px.bar(
        summary, x="venue", y="avg_route_prob", title="Avg Route Prob by Venue"
    )
    figs.append(fig_prob)

    dfx = df.copy()
    if "filled" in dfx.columns:
        dfx["filled_label"] = dfx["filled"].map({0: "No", 1: "Yes"})
    fig_scatter = px.scatter(
        dfx,
        x="route_prob",
        y="filled_qty",
        color="venue",
        symbol="filled_label" if "filled_label" in dfx.columns else None,
        title="Route Prob vs Filled Qty",
        hover_data=["order_id", "price", "qty"],
    )
    figs.append(fig_scatter)

    from plotly.subplots import make_subplots

    rows = len(figs)
    combined = make_subplots(
        rows=rows, cols=1, subplot_titles=[f.layout.title.text for f in figs]
    )
    for i, f in enumerate(figs, start=1):
        for tr in f.data:
            combined.add_trace(tr, row=i, col=1)
    combined.update_layout(
        height=350 * rows, showlegend=True, title_text="Final Fill Quality Report"
    )

    combined.write_html(out_html, include_plotlyjs="cdn")
    return out_html
