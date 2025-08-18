from __future__ import annotations

import os
import pandas as pd

# Try to import Plotly, but gracefully fallback if it's not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    px = None
    go = None
    make_subplots = None


def build_dashboard(csv_path: str, out_html: str = "dashboard.html") -> str:
    """
    Build a dashboard from the final fill-quality CSV.
    If Plotly isn't installed, write a minimal HTML summary instead.
    """
    df = pd.read_csv(csv_path)

    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Coerce dtypes
    for col in ("filled", "filled_qty", "queue_ahead", "agg_consumed", "price", "qty", "route_prob"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Summary by venue (handle case when 'venue' might be absent)
    if "venue" in df.columns:
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
    else:
        summary = pd.DataFrame(columns=["venue", "filled_rate", "orders", "avg_fill_qty", "avg_queue_ahead", "avg_route_prob"])

    if px is None or go is None or make_subplots is None:
        # Minimal fallback HTML
        html = "<html><head><title>Final Fill Quality Report</title></head><body>"
        html += "<h2>Summary by Venue</h2>"
        if len(summary):
            html += summary.to_html(index=False)
        else:
            html += "<p>No venue summary available.</p>"
        html += "<h2>Sample Fills (first 200)</h2>"
        if len(df):
            html += df.head(200).to_html(index=False)
        else:
            html += "<p>No fills to display.</p>"
        html += "</body></html>"
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        return out_html

    # Plotly version
    figs = []
    if len(summary):
        fig_filled = px.bar(summary, x="venue", y="filled_rate", title="Filled Rate by Venue", text="filled_rate")
        fig_filled.update_layout(yaxis_tickformat=".0%")
        figs.append(fig_filled)

        fig_orders = px.bar(summary, x="venue", y="orders", title="Orders by Venue", text="orders")
        figs.append(fig_orders)

        fig_prob = px.bar(summary, x="venue", y="avg_route_prob", title="Avg Route Prob by Venue")
        figs.append(fig_prob)

    dfx = df.copy()
    if "filled" in dfx.columns:
        dfx["filled_label"] = dfx["filled"].map({0: "No", 1: "Yes"})
    fig_scatter = px.scatter(
        dfx,
        x="route_prob" if "route_prob" in dfx.columns else None,
        y="filled_qty" if "filled_qty" in dfx.columns else None,
        color="venue" if "venue" in dfx.columns else None,
        symbol="filled_label" if "filled_label" in dfx.columns else None,
        title="Route Prob vs Filled Qty",
        hover_data=[c for c in ["order_id", "price", "qty"] if c in dfx.columns],
    )
    figs.append(fig_scatter)

    rows = len(figs)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=[f.layout.title.text for f in figs])
    for i, f in enumerate(figs, start=1):
        for tr in f.data:
            fig.add_trace(tr, row=i, col=1)
    fig.update_layout(height=350 * rows, showlegend=True, title_text="Final Fill Quality Report")

    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html
