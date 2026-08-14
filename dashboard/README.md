# Streamlit dashboard

Run the dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

The app reads generated report files from `reports/universe/`. If the reports are missing, run:

```bash
python -m scripts.run_walk_forward_universe
```
