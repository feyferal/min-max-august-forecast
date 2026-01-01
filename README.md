# min-max-august-forecast

CLI script that prints a table with **last 5 Augusts + 2026 forecast** for:
- `max_tmax` (August max of daily max temperature)
- `min_tmin` (August min of daily min temperature)

## Setup

```bash
python -m venv .venv
```

Activate virtual environment (Windows / PowerShell):
```
.\.venv\Scripts\activate
```
Activate virtual environment (macOS / Linux):
```
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

RUN

Default run (uses cached CSV if exists):
```
python -m src.main
```
Force refresh data from Open-Meteo and overwrite the cache:
```
python -m src.main --refresh
```
Use only last N years for training(default=15):
```
python -m src.main --window-years 20
python -m src.main --window-years 15
python -m src.main --window-years 10
```
Save the printed table to a file:
```
python -m src.main --out outputs/result.txt
```
