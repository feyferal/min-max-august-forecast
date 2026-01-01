# min-max-august-forecast

CLI script that prints a table with **last 5 Augusts + 2026 forecast** for:
- `max_tmax` — August maximum of daily maximum temperature
- `min_tmin` — August minimum of daily minimum temperature

The script automatically downloads historical daily weather data from Open-Meteo (or uses cached data if available), aggregates August statistics, and forecasts extreme temperatures for August 2026.

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

### Default run (uses cached CSV if exists)
If the cache file does not exist, it will be downloaded automatically.

```bash
python -m src.main
```

### Force refresh data from Open-Meteo and overwrite the cache
```bash
python -m src.main --refresh
```

### Use only last N years for training (default: 15)
```bash
python -m src.main --window-years 20
python -m src.main --window-years 15
python -m src.main --window-years 10
```

### Save the printed table to a file
```bash
python -m src.main --out outputs/result.txt
```

---

## CLI options

- `--data` (default: `data/raw/madrid_daily.csv`)  
  Path to cached daily data CSV (relative to project root)

- `--refresh`  
  Fetch fresh data from Open-Meteo even if cache exists

- `--window-years` (default: `15`)  
  Number of recent years used for training

- `--log-level` (default: `INFO`)  
  Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

- `--out`  
  Save printed table to a text file

---

## Examples

```bash
python -m src.main --log-level DEBUG
python -m src.main --data data/raw/madrid_daily.csv
python -m src.main --refresh --window-years 10 --out outputs/result.txt
```

---

## Output

The script prints a table with:
- the last 5 historical Augusts
- the forecasted extreme temperatures for August 2026

An example output file is available in the `outputs/` directory.
