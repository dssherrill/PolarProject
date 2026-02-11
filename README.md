# PolarProject
Displays performance charts derived from a glider's polar curve

Calculates speed-to-fly by solving Reichmann's Speed-to-Fly (STF) equation (see 
"MacCready speed to fly theory.pdf" in this project).

## Active Deployment
https://polars.up.railway.app/

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

**For detailed Railway deployment instructions, see [RAILWAY.md](RAILWAY.md)**

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

Polar Explorer is an older program that does similar things, but goes well beyond just speed-to-fly, with calculations for final glide and optimal thermalling speed and bank angle.  It also adjusts for atmospherics (temperature, pressure, altitude).
https://www.trimill.com/CuSoft/PolarExplorer/index.htm

