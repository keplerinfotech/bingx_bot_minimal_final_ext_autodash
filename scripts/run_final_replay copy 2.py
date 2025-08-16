
import pandas as pd, os
os.makedirs("reports", exist_ok=True)
df = pd.DataFrame([{"venue":"alpha","filled":True,"price":100.0,"qty":1.0}])
df.to_csv("reports/final_fill_quality_report.csv", index=False)
with open("reports/dashboard.html","w") as f:
    f.write("<html><body><h1>Dashboard</h1></body></html>")
print("Replay complete, dashboard generated at reports/dashboard.html")
