import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def build_dashboard(csv_path: str, html_out: str):
    df = pd.read_csv(csv_path)
    figs = []

    # Venue-level fill rate vs. route_prob
    venue_stats = df.groupby("venue").agg(fill_rate=("filled", "mean"), avg_route_prob=("route_prob", "mean")).reset_index()
    fig1 = px.scatter(venue_stats, x="avg_route_prob", y="fill_rate", color="venue",
                      title="Venue Fill Rate vs Route Probability",
                      labels={"avg_route_prob":"Avg Route Prob", "fill_rate":"Fill Rate"})
    figs.append(fig1)

    # Queue-ahead histogram
    fig2 = px.histogram(df, x="queue_ahead", color="venue", nbins=30, title="Queue Ahead Distribution")
    figs.append(fig2)

    # Latency impact
    if "latency_ms" in df.columns:
        fig3 = px.scatter(df, x="latency_ms", y="filled_qty", color="venue",
                          title="Latency vs Filled Quantity",
                          labels={"latency_ms":"Latency (ms)", "filled_qty":"Filled Qty"})
        figs.append(fig3)

    # Combine into HTML
    html_parts = []
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    with open(html_out, "w") as f:
        f.write("<html><head><title>Fill Quality Dashboard</title></head><body>")
        for part in html_parts:
            f.write(part)
            f.write("<hr>")
        f.write("</body></html>")
    print(f"Dashboard written to {html_out}")

if __name__ == "__main__":
    build_dashboard("final_fill_quality_report.csv", "dashboard.html")
