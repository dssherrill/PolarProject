# PolarProject AI Coding Agent Instructions

## Project Overview
PolarProject calculates and visualizes glider performance based on **polar curves** (speed vs. sink rate). It solves the **MacCready Speed-to-Fly equation** to find optimal cruising speeds given thermal strength and other flight conditions.

- **Domain**: Soaring/Gliding aerodynamics
- **Key Reference**: "MacCready speed to fly theory.pdf" (included)
- **Primary Goal**: Display polar charts and compute speed-to-fly recommendations

## Architecture

### Core Components

**[polar_calc.py](polar_calc.py)** - Calculation engine  
- `Polar` class: Main domain logic
  - **Inputs**: Glider specs, polynomial degree, wind conditions, MacCready value
  - **Workflow**: Load CSV polar → Fit polynomial → Solve STF equation using `fsolve()`
  - **Key methods**: `goal_function()`, `sink()`, `MacCready()`
  - Supports two goal functions: "Reichmann" (minimum time) and "Test" (max avg speed)

**[polar_ui.py](polar_ui.py)** - Dash web UI (536 lines)
- `load_polar()` function: Orchestrates calculation and returns DataFrame for display
- Plotly graphs with dual-axis support (speed vs. sink, speed-to-fly curves)
- AG Grid table for MacCready results
- Responsive Flexbox layout with Dash Bootstrap Components

**[units.py](units.py)** - Centralized unit registry  
- **Critical**: Single `ureg` instance must be imported globally (not created locally)
- Used throughout: `ureg.mps`, `ureg.kph`, `ureg.kg`, etc.
- Enables Pint PintArray columns in DataFrames

**[datafiles/gliderInfo.json](datafiles/gliderInfo.json)** - Glider configuration  
- Defines available gliders: ASK 21, ASW 28, Duo Discus T, SGS 1-26
- Metadata: `referenceWeight`, `emptyWeight`, `polarFileName`, `fitStart/fitEnd`

**[datafiles/\*.csv](datafiles/)** - WebPlotDigitizer exports  
- Format: Speed, Sink (any and all sensible units allowed)
- Created via [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer)
- Note: 1 CSV can represent multiple glider variants (e.g., ASK 21 K1 vs K2)

### Data Flow
```text
gliderInfo.json → Glider metadata
    ↓
polar_calc.Polar() → Load CSV → Fit polynomial → Solve STF
    ↓
polar_ui.load_polar() → Create display DataFrame
    ↓
Plotly/AG Grid → User visualization
```

## Unit Handling (Pint/Pint-Pandas)

**Pattern**: All calculations use SI units internally; conversions happen at UI boundary.

```python
# In polar_calc.py:
from units import ureg
speed = (df.iloc[:,0].to_numpy() * ureg.kph).to('mps')  # kph → m/s
sink = df.iloc[:,1].to_numpy() * ureg.mps

# In polar_ui.py:
METRIC_UNITS = {'Speed': ureg('kph'), 'Sink': ureg('m/s')}
US_UNITS = {'Speed': ureg('knots'), 'Sink': ureg('knots')}
# User selects; conversions applied at display time
```

**Important**: Do NOT import Pint directly without `units.ureg`—use the shared registry.

## Weight Adjustment Factor

Glider performance scales with weight. Key calculation (in `Polar.__init__`):
```python
weight_factor = √(actual_weight / reference_weight)
# Adjusts sink rate: sink_adjusted = weight_factor * sink_nominal + vertical_wind
```

Reference weight from JSON; empty weight + pilot weight = actual weight.

## Polynomial Fitting

- Default degree: 5 (tunable via UI)
- Uses NumPy `Polynomial` class (not deprecated)
- Fit range: `fitStart`/`fitEnd` from gliderInfo.json (e.g., 50–200 kph)
- Extrapolation beyond range may be unreliable

## Key Dependencies & Conventions

- **Dash** (UI framework)
- **Plotly** (graphing)
- **Pint + Pint-Pandas** (units)
- **NumPy/SciPy** (numerical: polynomial fitting, `fsolve()`)
- **Pandas** (DataFrames + PintArray columns)
- **Dash AG Grid** (data tables)
- **Bootstrap CSS** (responsive styling)

## Development Workflow

### Running the UI
```bash
# Install dependencies
pip install -r requirements.txt

# Activate venv (Windows)
.venv\Scripts\Activate.ps1

# Run Dash app
python polar_ui.py
```
Serves at `http://127.0.0.1:8050/`

### Testing Polar Calculations
Use `pint_*_demo.py` files for unit/dataframe examples (exploration scripts, not production).

### Adding a New Glider
1. Export polar from WebPlotDigitizer → `datafiles/*.csv`
2. Add entry to `datafiles/gliderInfo.json` with metadata
3. UI dropdown auto-updates

## Common Patterns & Gotchas

- **Units must flow through pipelines**: Don't strip `.magnitude` unless absolutely necessary; let Pint track them.
- **Weight factor applies non-linearly**: Affects both sink rate and all derived metrics.
- **CSV format strict**: Speed (col 1), Sink (col 2). No header row.
- **Goal functions differ**: Reichmann minimizes time to reach a goal; "Mine" maximizes average speed accounting for wind.
- **fsolve() initial guess matters**: Currently 50 knots; may need adjustment for edge cases.

## File Structure Notes

- `test` directory does NOT exist (no formal test suite)
- `glider.py` appears deprecated; core logic now in `polar_calc.Polar`
- `*_demo.py` files are exploration/learning aids
- `.xlsx` files are manual analysis outputs (not part of build)
