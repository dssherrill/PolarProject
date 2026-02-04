# PolarProject
Displays performance charts derived from a glider's polar curve

Calculates speed-to-fly by solving Reichmann's Speed-to-Fly (STF) equation (see 
"MacCready speed to fly theory.pdf" in this project).

## Requirements

- Python 3.12 or higher

## Installation

### For Local Development

Using pyproject.toml (recommended for modern Python packaging):
```bash
pip install -e .
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### For Deployment (Railway, Heroku, etc.)

The `requirements.txt` file is included for deployment platforms that rely on it. Railway and similar platforms will automatically detect and use it.

## Running the Application

```bash
python polar_ui.py
```

Then open your browser to http://127.0.0.1:8050/

### Production Deployment

For production deployments, use gunicorn (already included in dependencies):
```bash
gunicorn polar_ui:server
```

## Resources:

John Cochrane articles
https://www.johnhcochrane.com/about/soaring

Polar Explorer is an older program that does similar things, but goes well beyond just speed-to-fly.
https://www.trimill.com/CuSoft/PolarExplorer/index.htm

