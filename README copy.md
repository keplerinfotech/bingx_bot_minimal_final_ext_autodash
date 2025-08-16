# Bingx Bot Minimal - Final Extended

## New Modules
- `scripts/convert_recorder_multi.py`: Convert multiple-venue JSONL recorders to L2 + trades parquet format.
- `scripts/dashboard.py`: Generate an interactive HTML dashboard from `final_fill_quality_report.csv`.

## Usage
1. Convert your multi-venue recorder files:
```
python scripts/convert_recorder_multi.py
```
(Edit the recorder_paths dict in the script.)

2. Run your replay to generate `final_fill_quality_report.csv`.

3. Build dashboard:
```
python scripts/dashboard.py
```
View `dashboard.html` in your browser.
