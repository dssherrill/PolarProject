import math
import os
import re

import numpy as np
import dash
from dash import Dash, dcc, html, Input, Output, State, callback, set_props

import dash_bootstrap_components as dbc
import dash.exceptions

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash_ag_grid as dag

import polar_calc
import glider
import pint_pandas

import logging

# Get access to the one-and-only UnitsRegistry instance
from units import ureg

PA_ = pint_pandas.PintArray
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid filesystem characters.

    Removes or replaces characters that are invalid in Windows/Unix filenames:
    / \ : * ? " < > |

    Also trims excessive whitespace and limits length to 255 characters.

    Args:
        filename: The filename string to sanitize

    Returns:
        Sanitized filename string safe for use in filesystem operations
    """
    # Remove invalid filesystem characters: / \ : * ? " < > |
    sanitized = re.sub(r'[/\\:*?"<>|]', "", filename)
    # Replace multiple consecutive spaces with a single space
    sanitized = re.sub(r"\s+", " ", sanitized)
    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()
    # Limit to 255 characters (filesystem limit)
    sanitized = sanitized[:255]
    return sanitized


# Production mode flag and glider cache
_production_mode = True
_glider_cache = {}  # Cache for Glider instances to avoid duplicate CSV parsing

# Read list of available polars from a file
df_glider_info = pd.read_json("datafiles/gliderInfo.json")
DEFAULT_GLIDER_NAME = "ASW 28"

# Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
# app = Dash(external_stylesheets=[dbc.themes.SLATE])

# Expose server for deployment platforms (Railway, Heroku, etc.)
server = app.server

# polynomial fit degree (order)
DEFAULT_POLYNOMIAL_DEGREE = 5

# display options
METRIC_UNITS = {
    "Speed": ureg("kph"),
    "Sink": ureg("m/s"),
    "Weight": ureg("kg"),
    "Wing Area": ureg("m**2"),
    "Pressure": ureg("kg/m**2"),
}
US_UNITS = {
    "Speed": ureg("knots"),
    "Sink": ureg("knots"),
    "Weight": ureg("lbs"),
    "Wing Area": ureg("ft**2"),
    "Pressure": ureg("lbs/ft**2"),
}
UNIT_CHOICES = {"Metric": METRIC_UNITS, "US": US_UNITS}


def get_cached_glider(glider_name, current_glider_info) -> glider.Glider:
    """
    Get or create a cached Glider instance to avoid duplicate CSV parsing.

    Parameters:
        glider_name: Name of the glider
        current_glider_info: DataFrame row with glider information

    Returns:
        glider.Glider: Cached or newly created Glider instance
    """
    global _glider_cache

    if glider_name not in _glider_cache:
        _glider_cache[glider_name] = glider.Glider(current_glider_info)

    return _glider_cache[glider_name]


def load_polar(
    current_glider: glider.Glider,
    degree,
    goal_function,
    v_air_horiz,
    v_air_vert,
    pilot_weight,
):
    """
    Create a Polar model for the given glider configured with the specified fit degree, goal function, airmass speeds, and pilot weight.

    Parameters:
        current_glider (glider.Glider): Glider definition and performance data to build the polar from.
        degree (int): Polynomial degree used to fit the polar curve (must be >= 2).
        goal_function (str): Identifier of the optimization goal used when constructing the polar (e.g., 'Reichmann', 'Test').
        v_air_horiz (float): Horizontal airmass speed to apply to the polar, in meters per second.
        v_air_vert (float): Vertical airmass (sink or lift) speed to apply to the polar, in meters per second.
        pilot_weight (float | pint.Quantity): Pilot weight to include in the polar; may be a plain number (assumed kilograms) or a quantity with units.

    Returns:
        polar_calc.Polar: A Polar object representing the fitted polar model for the provided glider and conditions.
    """
    current_polar = polar_calc.Polar(
        current_glider, degree, goal_function, v_air_horiz, v_air_vert, pilot_weight
    )
    return current_polar


# dummy data to setup the AG Grid as a MacCready table
initial_mc_data = pd.DataFrame({"MC": [0], "STF": [0], "Vavg": [0], "L/D": [0]})
initial_glider_data = pd.DataFrame(
    {
        "Label": [
            "Reference Weight",
            "Empty Weight",
            "Pilot+Ballast Weight",
            "Gross Weight",
            "Wing Loading",
        ],
        "Metric": ["-- kg", "-- kg", "-- kg", "-- kg", "-- kg/m²"],
        "US": ["-- lbs", "-- lbs", "-- lbs", "-- lbs", "-- lb/ft²"],
    },
)

# Format the data for the dcc.Dropdown 'options' property
# It requires a list of dictionaries with 'label' (what the user sees)
# and 'value' (the actual data used internally) keys.
dropdown_options = [
    {"label": row["name"], "value": row["name"]}
    for index, row in df_glider_info.iterrows()
]

# Styles for Flexbox container and items
container_style = {
    "display": "flex",
    "flex-wrap": "wrap",  # Allows items to stack on small screens
    # "gap": "20px",       # Space between items
    # "padding": "20px"
}

# Responsive item: occupies half width on large screens, full width on small
item_style = {
    "flex": "0 0 600px",  # grow=1, shrink=1, base_width=450px
    "min-width": "450px",
    "min-height": "550px",
}

graph_style = {
    "flex": "1 1 600px",  # grow=1, shrink=1, base_width=800px
    # "min-width": "300px",
    "min-height": "500px",
}

full_width_class = "d-flex w-100 justify-content-left bg-light"  # p-3"


def add_graph(graph_id):
    """Create a Graph component with the specified ID."""
    return html.Div(
        [
            dcc.Graph(
                figure={},
                id=graph_id,
            )
        ],
    )


def add_mc_aggrid():
    """Create an AG Grid component with the specified ID."""
    return html.Div(
        [
            dag.AgGrid(
                columnSize="autoSize",
                dashGridOptions={"domLayout": "autoHeight"},
                style={"width": "100%"},
                id="mcAgGrid",
            )
        ]
    )


def add_compare_controls():
    """
    Create a Dash HTML container with controls to save and clear Speed-to-Fly (STF) results and to choose the comparison display mode.

    The container includes:
    - a "Save for Comparison" button,
    - a "Clear Comparison" button,
    - a radio control with options "Raw" and "Subtracted" (default "Subtracted") to select how saved comparisons are displayed.

    Returns:
        html.Div: A Dash HTML container with the comparison buttons and radio items.
    """
    return html.Div(
        [
            dbc.Stack(
                [
                    dbc.Button(
                        "Save for Comparison",
                        id="save-comparison-button",
                        className="m-2",
                        # color="secondary",
                    ),
                    dbc.Button(
                        "Clear Comparison",
                        id="clear-comparison-button",
                        className="m-2",
                        # color="secondary",
                        disabled=True,
                    ),
                    dbc.Label(
                        "Display:",
                        className="text-primary fs-4",
                    ),
                    dbc.RadioItems(
                        options=["Raw", "Subtracted"],
                        value="Subtracted",
                        inline=False,
                        id="radio-subtract-compare",
                        persistence=True,
                        persistence_type="local",
                    ),
                    dbc.Label(
                        "Compare:",
                        className="text-primary fs-4",
                    ),
                    dbc.RadioItems(
                        options=[
                            {"label": "Speed-to-Fly", "value": "STF"},
                            {"label": "Average Speed", "value": "Vavg"},
                        ],
                        value="STF",
                        inline=False,
                        id="radio-compare-metric",
                        persistence=True,
                        persistence_type="local",
                    ),
                ],
                direction="horizontal",
                className="hstack gap-3 m-3",
            ),
        ],
        className="d-flex align-items-start m-3",
    )


def add_compare_options():
    """
    Render the comparison option controls for STF plotting mode selection.

    Returns:
        tuple: Two Bootstrap column components:
            - A `dbc.Col` containing a label for the option selector.
            - A `dbc.Col` containing a `dbc.RadioItems` with choices "Raw" and "Subtracted" (default "Subtracted") and local persistence.
    """
    # return html.Div(
    #     [
    return (
        dbc.Col(
            dbc.Label(
                "Option:",
                className="text-primary fs-4",
            ),
            md=3,
        ),
        dbc.Col(
            dbc.RadioItems(
                options=["Raw", "Subtracted"],
                value="Subtracted",
                inline=False,
                id="radio-subtract-compare",
                persistence=True,
                persistence_type="local",
            ),
            md=3,
        ),
        #        ],
    )


# App layout
app.layout = dbc.Container(
    [
        dcc.Store(id="user-data-store", storage_type="localStorage"),
        dcc.Store(id="working-data-store", storage_type="localStorage"),
        dcc.Store(id="df-out-store", storage_type="localStorage"),
        html.Div(
            "Glider Polar Analysis Tool",
            className="text-primary fs-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="main-title",
                            className="text-primary fs-3",
                        )
                    ],
                    width=12,
                )
            ],
            className=full_width_class,
        ),
        dbc.Row(
            [
                # Input controls group
                dbc.Col(
                    [
                        html.Label("Select a glider:"),
                        dcc.Dropdown(
                            id="glider-dropdown",
                            options=dropdown_options,
                            value=DEFAULT_GLIDER_NAME,  # dropdown_options[0]['value'] if dropdown_options else None, # Set a default value if options exist
                            placeholder="Select a glider...",
                            persistence=True,
                            persistence_type="local",
                            style={"width": "100%"},
                        ),
                        dag.AgGrid(
                            rowData=initial_glider_data.to_dict("records"),
                            columnDefs=[
                                {"field": "Label"},
                                {"field": "Metric"},
                                {"field": "US"},
                            ],
                            columnSize="sizeToFit",
                            dashGridOptions={"domLayout": "autoHeight"},
                            # dashGridOptions={"pagination": False},
                            style={"width": "100%"},
                            id="glider_AgGrid",
                            # className=full_width_class,
                        ),
                        # --- Unit selection
                        html.Div(
                            [
                                dbc.Label(
                                    "Units:",
                                    className="text-primary fs-4",
                                ),
                                dbc.RadioItems(
                                    options=["Metric", "US"],
                                    value="Metric",
                                    inline=False,
                                    id="radio-units",
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ],
                        ),
                        # --- Input choices
                        dbc.Label(
                            "Input Option:",
                            html_for="radio-weight-or-loading",
                            id="input-choice-prompt",
                            style={
                                "margin-top": "15px",
                                "width": 450,
                            },
                            className="text-primary fs-4",
                        ),
                        dbc.RadioItems(
                            options=[
                                "Pilot Weight",
                                "Wing Loading",
                            ],
                            value="Wing Loading",
                            inline=False,
                            id="radio-weight-or-loading",
                            persistence=True,
                            persistence_type="local",
                        ),
                        #
                        #  One of the next two inputs, either pilot weight or wing loading, will be shown
                        # but not both.
                        #
                        html.Div(
                            [
                                # --- pilot weight
                                dbc.Label(
                                    "Pilot + Ballast weight (kg):",
                                    html_for="pilot-weight-input",
                                    id="pilot-weight-label",
                                    style={
                                        "margin-top": "15px",
                                        "width": 450,
                                    },
                                ),
                                dcc.Input(
                                    id="pilot-weight-input",
                                    type="number",
                                    placeholder="100",
                                    style={"margin-end": 100, "width": 450},
                                    debounce=0.75,
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ],
                            className="mb-3",
                            id="pilot-weight-row",
                        ),
                        html.Div(
                            [
                                # --- wing loading
                                dbc.Label(
                                    "Wing Loading (kg/m²)",
                                    html_for="wing-loading-input",
                                    id="wing-loading-label",
                                    style={
                                        "margin-top": "15px",
                                        "width": 450,
                                    },
                                ),
                                dcc.Input(
                                    id="wing-loading-input",
                                    type="number",
                                    placeholder="100",
                                    style={"margin-end": 100, "width": 450},
                                    debounce=0.75,
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ],
                            className="mb-3",
                            id="wing-loading-row",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        # --- Polynomial degree
                        dbc.Row(
                            [
                                dbc.Label(
                                    "Polynomial degree:",
                                    html_for="poly-degree",
                                    className="text-start",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Input(
                                            id="poly-degree",
                                            type="number",
                                            placeholder=DEFAULT_POLYNOMIAL_DEGREE,
                                            min=2,
                                            className="text-start",
                                            debounce=0.75,
                                        ),  # style={"width": "1"}),
                                    ]
                                ),  # width=10),
                            ]
                        ),
                        # Statistics output group
                        dbc.Row(
                            [
                                html.Br(),
                                dcc.Markdown(
                                    "Statistics",
                                    style={
                                        "whiteSpace": "pre-line",
                                        "fontFamily": "Courier, serif",
                                    },
                                    id="statistics",
                                    mathjax=True,  # Enable LaTeX rendering
                                    dangerously_allow_html=True,
                                    className="text-start",
                                ),  # enable html display without purify
                            ],
                            # width={"size": 2, "offset": 1},
                            # className="mt-3 mb-3",
                            className=full_width_class,
                        ),
                    ],
                    # width=5,
                ),
                # dbc.Col(
                #     html.Div("Additional Info"),
                #     # width=2
                # ),
            ],
            className=full_width_class,
        ),
        # --- Graphs
        html.Br(),
        dbc.Row(
            [add_compare_controls()],
            className=full_width_class,
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    add_graph("graph-polar"),
                    md=4,
                ),
                dbc.Col(
                    add_graph("graph-stf"),
                    md=4,
                ),
                dbc.Col(
                    add_mc_aggrid(),
                    md=4,
                ),
            ],
            className=full_width_class,
            style={"flex-wrap": "wrap"},
        ),
        html.Br(),
        dbc.Row(
            [
                html.Div(
                    "Airmass Movement",
                    # className="text-primary fs-3",
                    className="d-none",  # TODO: enable later
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Stack(
                    [
                        dbc.Label(
                            "Horizontal speed",
                            id="horizontal-speed-label",
                        ),
                        dcc.Input(
                            id="airmass-horizontal-speed",
                            type="number",
                            value=0,
                            placeholder="0",
                            debounce=0.75,
                            style={"width": "450px"},
                        ),
                        dbc.Label(
                            "Vertical speed",
                            id="vertical-speed-label",
                            html_for="airmass-vertical-speed",
                        ),
                        dcc.Input(
                            id="airmass-vertical-speed",
                            type="number",
                            value=0,
                            placeholder="0",
                            debounce=0.75,
                            style={"width": "450px"},
                        ),
                    ],
                    style={"width": "450px", "gap": "10px"},
                ),
            ],
            style={
                "flex": "1 1 100%",
            },
            className="d-none",  # TODO: enable later
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            "Debug Options",
                            className="text-primary fs-3 mt-5",
                        ),
                        dbc.Switch(
                            id="toggle-switch-dump",
                            label="Dump file",
                            value=False,  # Initial state
                            className="ms-3",
                            disabled=True,  # enabled only in development, never in production
                        ),
                        dbc.Switch(
                            id="toggle-switch-debug",
                            label="Include debug graphs (Residuals, Goal Function and Solver Result)",
                            value=False,  # Initial state
                            className="ms-3",
                        ),
                        dbc.Label(
                            "MacCready Value for Goal Function debug graph",
                            html_for="maccready-input",
                            className="w-2",
                        ),
                        html.Br(),
                        dcc.Input(
                            id="maccready-input",
                            type="number",
                            placeholder="0",
                            style={"width": 450},
                        ),
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Solution",
                                            className="text-primary fs-3 mt-5",
                                        ),
                                        dbc.RadioItems(
                                            options=["Reichmann", "Test"],
                                            value="Reichmann",
                                            inline=False,
                                            id="radio-goal",
                                        ),
                                    ],
                                ),
                            ],
                            className="d-none mb-3",  # TODO: consider removing
                        ),
                    ],
                    md=4,
                ),
            ],
            style={
                "flex": "1 1 100%",
            },
        ),  # "margin-top": "20px"}),
    ],
    style=container_style,
    # className="flex-wrap flex-lg-nowrap",
    className=full_width_class,
    fluid=True,
)


##################################################################
# Handle unit system changes and update labels accordingly
# Also capture pilot weight and airmass speeds into the data store
@callback(
    Output(component_id="glider_AgGrid", component_property="rowData"),
    Output(component_id="horizontal-speed-label", component_property="children"),
    Output(component_id="vertical-speed-label", component_property="children"),
    Output(component_id="pilot-weight-label", component_property="children"),
    Output(component_id="pilot-weight-input", component_property="placeholder"),
    Output(component_id="pilot-weight-input", component_property="value"),
    Output(component_id="wing-loading-label", component_property="children"),
    Output(component_id="wing-loading-input", component_property="placeholder"),
    Output(component_id="wing-loading-input", component_property="value"),
    Output(component_id="wing-loading-input", component_property="min"),
    Output(component_id="airmass-horizontal-speed", component_property="value"),
    Output(component_id="airmass-vertical-speed", component_property="value"),
    Output("user-data-store", "data"),
    Output("working-data-store", "data"),
    Input(component_id="radio-units", component_property="value"),
    Input(component_id="radio-weight-or-loading", component_property="value"),
    Input(component_id="glider-dropdown", component_property="value"),
    Input(component_id="pilot-weight-input", component_property="value"),
    Input(component_id="wing-loading-input", component_property="value"),
    Input(component_id="airmass-horizontal-speed", component_property="value"),
    Input(component_id="airmass-vertical-speed", component_property="value"),
    State(component_id="user-data-store", component_property="data"),
    #    prevent_initial_call=True
)
def process_unit_change(
    units,
    weight_or_loading,
    glider_name,
    pilot_weight_in,
    wing_loading_in,
    v_air_horiz_in,
    v_air_vert_in,
    data,
):
    """
    Process unit changes and update glider parameters based on user input.
    Handles unit system selection and manages pilot weight/wing loading inputs,
    converting between different unit systems and calculating derived values.
    Args:
        units (str): Selected unit system (e.g., 'Metric', 'US').
        weight_or_loading (str): User selection between 'Pilot Weight' or 'Wing Loading' input mode.
        glider_name (str): Name of the selected glider.
        pilot_weight_in (float): Pilot weight input value in selected units.
        wing_loading_in (float): Wing loading input value in selected units.
        v_air_horiz_in (float): Horizontal airmass speed input in selected units.
        v_air_vert_in (float): Vertical airmass speed input in selected units.
        data (dict): Stored state data containing pilot_weight, wing_loading, v_air_horiz, v_air_vert.
    Returns:
        tuple: A tuple containing:
            - glider_data (list): DataFrame records with glider weight and loading specifications.
            - horizontal_speed_label (str): Label for horizontal speed input field.
            - vertical_speed_label (str): Label for vertical speed input field.
            - pilot_weight_label (str): Label for pilot weight input field.
            - pilot_weight_placeholder (str): Placeholder text for pilot weight input.
            - pilot_weight_value (float): Current pilot weight value in selected units.
            - wing_loading_label (str): Label for wing loading input field.
            - wing_loading_placeholder (str): Placeholder text for wing loading input.
            - wing_loading_value (float): Current wing loading value in selected units.
            - min_wing_loading_display (float): Minimum wing loading constraint.
            - v_air_horiz (float): Horizontal airmass speed in m/s.
            - v_air_vert (float): Vertical airmass speed in m/s.
            - data_store (dict): Updated state dictionary with current values.
    Raises:
        dash.exceptions.PreventUpdate: If no glider is selected or glider info is empty.
    """
    logger.info(
        f"process_unit_change called with units={units}, weight_or_loading={weight_or_loading}, glider_name={glider_name}, pilot_weight_in={pilot_weight_in}, wing_loading_in={wing_loading_in}, v_air_horiz_in={v_air_horiz_in}, v_air_vert_in={v_air_vert_in}, data={data}"
    )
    logger.debug(f"process_unit_change called _production_mode={_production_mode}")

    # Show the selected input box; hide the other input box
    if weight_or_loading == "Pilot Weight":
        set_props("pilot-weight-row", {"className": "mb-3"})
        set_props("wing-loading-row", {"className": "d-none"})
    else:
        set_props("wing-loading-row", {"className": "mb-3"})
        set_props("pilot-weight-row", {"className": "d-none"})

    # setup units
    selected_units = UNIT_CHOICES[units]
    weight_units = selected_units["Weight"]
    speed_units = selected_units["Speed"]
    sink_units = selected_units["Sink"]
    pressure_units = selected_units["Pressure"]

    # Use default glider if none selected
    glider_name = glider_name if glider_name else DEFAULT_GLIDER_NAME
    current_glider_info = df_glider_info[df_glider_info["name"] == glider_name]

    # Just return if glider data is not available
    if current_glider_info.empty:
        raise dash.exceptions.PreventUpdate

    # Get or create cached Glider instance to avoid duplicate CSV parsing
    current_glider = get_cached_glider(glider_name, current_glider_info)

    # Fetch stored values or initialize to None
    pilot_weight = data.get("pilot_weight") if data else None
    wing_loading = data.get("wing_loading") if data else None
    v_air_horiz = data.get("v_air_horiz") if data else 0.0
    v_air_vert = data.get("v_air_vert") if data else 0.0

    # Default to reference values if not set.
    # Reference values for weight and wing loading are those for the original the polar chart
    working_pilot_weight = (
        pilot_weight * ureg("kg")
        if pilot_weight is not None
        else current_glider.reference_pilot_weight()
    )
    working_wing_loading = (
        wing_loading * ureg("kg/m**2")
        if wing_loading is not None
        else current_glider.reference_wing_loading()
    )

    # Disable dump file option in production mode
    set_props("toggle-switch-dump", {"disabled": _production_mode})

    # Compute wing loading when pilot weight is zero (empty glider)
    # This is the minimum wing loading possible for this glider
    min_wing_loading = current_glider.empty_weight() / current_glider.wing_area()

    # this gets used in the wing loading input label and for the min attribute
    min_wing_loading_display = f"{min_wing_loading.to(pressure_units).magnitude:.1f}"

    # capture the pilot weight each time it changes
    if dash.ctx.triggered_id == "pilot-weight-input":
        logger.debug(f"{pilot_weight_in=}")
        # Process pilot weight input and compute wing loading to match
        if pilot_weight_in is None:
            pilot_weight = None
            working_pilot_weight = current_glider.reference_pilot_weight()
            wing_loading = None
            working_wing_loading = current_glider.reference_wing_loading()
        else:
            working_pilot_weight = (pilot_weight_in * weight_units).to("kg")
            pilot_weight = working_pilot_weight.magnitude

            # Compute wing loading from pilot weight
            gross_weight = current_glider.empty_weight() + working_pilot_weight
            working_wing_loading = gross_weight / current_glider.wing_area()
            wing_loading = working_wing_loading.magnitude

    # capture wing loading input value each time it changes
    if dash.ctx.triggered_id == "wing-loading-input":
        logger.debug(f"{wing_loading_in=}")
        # Process wing loading input and compute pilot weight to match
        if wing_loading_in is None:
            wing_loading = None
            working_wing_loading = current_glider.reference_wing_loading()
            pilot_weight = None
            working_pilot_weight = current_glider.reference_pilot_weight()
        else:
            working_wing_loading = (wing_loading_in * pressure_units).to("kg/m**2")
            if working_wing_loading < min_wing_loading:
                working_wing_loading = min_wing_loading  # adjust to minimum
            wing_loading = working_wing_loading.magnitude

            # Compute pilot weight from wing loading
            gross_weight = working_wing_loading * current_glider.wing_area()
            working_pilot_weight = gross_weight - current_glider.empty_weight()
            pilot_weight = working_pilot_weight.magnitude

    pilot_weight_label = f"Pilot + Ballast weight ({weight_units.units:~P}):"
    wing_loading_label = f"Wing Loading ({pressure_units.units:~P}), min {min_wing_loading_display} {pressure_units.units:~P}:"

    pilot_weight_placeholder = (
        f"{current_glider.reference_pilot_weight().to(weight_units).magnitude:.1f}"
    )
    wing_loading_placeholder = (
        f"{current_glider.reference_wing_loading().to(pressure_units).magnitude:.1f}"
    )

    gross_weight = current_glider.empty_weight() + working_pilot_weight

    # capture the horizontal airmass speed each time it changes
    if dash.ctx.triggered_id == "airmass-horizontal-speed":
        logger.debug("airmass-horizontal-speed clicked")

        if v_air_horiz_in is None:
            v_air_horiz = 0.0
        else:
            v_air_horiz = (v_air_horiz_in * speed_units).to("m/s").magnitude

    # capture the vertical airmass speed each time it changes
    if dash.ctx.triggered_id == "airmass-vertical-speed":
        logger.debug("airmass-vertical-speed clicked")
        if v_air_vert_in is None:
            v_air_vert = 0.0
        else:
            v_air_vert = (v_air_vert_in * sink_units).to("m/s").magnitude

    # Assemble glider data for display
    glider_data = pd.DataFrame(
        {
            "Label": [
                "Reference Weight",
                "Empty Weight",
                "Pilot+Ballast Weight",
                "Gross Weight",
                "Wing Loading",
            ],
            "Metric": [
                f"{current_glider.reference_weight():.1f~P}",
                f"{current_glider.empty_weight():.1f~P}",
                f"{working_pilot_weight:.1f~P}",
                f"{gross_weight:.1f~P}",
                f"{working_wing_loading:.1f~P}",
            ],
            "US": [
                f"{current_glider.reference_weight().to(US_UNITS['Weight']):.1f~P}",
                f"{current_glider.empty_weight().to(US_UNITS['Weight']):.1f~P}",
                f"{working_pilot_weight.to(US_UNITS['Weight']):.1f~P}",
                f"{gross_weight.to(US_UNITS['Weight']):.1f~P}",
                f"{(working_wing_loading.to(US_UNITS['Pressure'])):.1f~P}",
            ],
        }
    )

    return (
        glider_data.to_dict("records"),
        f"Horizontal speed ({selected_units['Speed'].units:~P}):",
        f"Vertical speed ({selected_units['Sink'].units:~P}):",
        pilot_weight_label,
        pilot_weight_placeholder,
        # Pilot weight display value
        (
            None
            if pilot_weight is None
            else round(working_pilot_weight.to(weight_units).magnitude, 1)
        ),
        wing_loading_label,
        wing_loading_placeholder,
        # Wing loading display value
        (
            None
            if wing_loading is None
            else round(working_wing_loading.to(pressure_units).magnitude, 1)
        ),
        float(min_wing_loading_display),
        (
            round((v_air_horiz * ureg("m/s")).to(speed_units).magnitude, 1)
            if v_air_horiz is not None
            else None
        ),
        (
            round((v_air_vert * ureg("m/s")).to(sink_units).magnitude, 1)
            if v_air_vert is not None
            else None
        ),
        {
            "pilot_weight": pilot_weight,
            "wing_loading": wing_loading,
            "v_air_horiz": v_air_horiz,
            "v_air_vert": v_air_vert,
        },
        {
            "pilot_weight": working_pilot_weight.magnitude,
            "wing_loading": working_wing_loading.magnitude,
            "v_air_horiz": v_air_horiz,
            "v_air_vert": v_air_vert,
        },
    )


##################################################################
@callback(
    Output(component_id="main-title", component_property="children"),
    Output(component_id="statistics", component_property="children"),
    Output(component_id="graph-polar", component_property="figure"),
    Output(component_id="graph-stf", component_property="figure"),
    Output(component_id="mcAgGrid", component_property="rowData"),
    Output(component_id="mcAgGrid", component_property="columnDefs"),
    Output(component_id="mcAgGrid", component_property="columnSize"),
    Output(component_id="poly-degree", component_property="value"),
    Output(component_id="clear-comparison-button", component_property="disabled"),
    Output("df-out-store", "data"),
    Input("working-data-store", "data"),
    Input(component_id="poly-degree", component_property="value"),
    Input(component_id="glider-dropdown", component_property="value"),
    Input(component_id="maccready-input", component_property="value"),
    Input(component_id="radio-goal", component_property="value"),
    Input(component_id="toggle-switch-debug", component_property="value"),
    Input(component_id="toggle-switch-dump", component_property="value"),
    Input(component_id="save-comparison-button", component_property="n_clicks"),
    Input(component_id="clear-comparison-button", component_property="n_clicks"),
    Input(component_id="radio-subtract-compare", component_property="value"),
    Input(component_id="radio-compare-metric", component_property="value"),
    State(component_id="radio-units", component_property="value"),
    State(component_id="radio-weight-or-loading", component_property="value"),
    State("df-out-store", "data"),
)
def update_graph(
    data,
    degree,
    glider_name,
    maccready,
    goal_function,
    show_debug_graphs,
    write_excel_file,
    save_comparison,
    clear_comparison,
    subtract_compare,
    compare_metric,
    units,
    weight_or_loading,
    df_out_data,
):
    """
    Compute and return the polar plot, Speed-to-Fly plot, MacCready table data, and other UI outputs based on the current glider, inputs, and stored state.

    Parameters:
        data (dict): Working-state values from storage; expected keys include 'pilot_weight', 'wing_loading', 'v_air_horiz', and 'v_air_vert' with SI magnitudes (m, m/s, kg, etc.) as appropriate.
        degree (int|None): Desired polynomial degree for the polar fit; a value less than 2 or None will be treated as 2.
        glider_name (str|None): Name of the selected glider; when falsy the application default glider is used.
        maccready (float|None): MacCready setting expressed in the currently selected sink units; None is treated as 0.0.
        goal_function (str): Identifier for the solver goal function used by the polar model.
        show_debug_graphs (bool): If True, include diagnostic traces (goal function, residuals, solver result) on the graphs.
        write_excel_file (bool): If True and the app is not in production mode, append saved STF results to an Excel file named "<sanitized glider name> stf.xlsx".
        save_comparison (bool): Trigger indicating the current STF should be saved into the comparison store.
        clear_comparison (bool): Trigger indicating previously saved comparison data should be cleared.
        subtract_compare (str): Comparison display mode; when equal to "Subtracted" and saved data exists, saved STF series are subtracted from the current STF for plotting.
        units (str): Unit system key, e.g. 'Metric' or 'US', used to choose display units for speed, sink, weight, and pressure.
        weight_or_loading (str): Mode string selecting whether legends/labels reference 'Pilot Weight' or 'Wing Loading'.
        df_out_data (dict|None): Serialized saved STF data from localStorage (or None) used to overlay or subtract previously saved series.

    Returns:
        tuple: (glider_name, polar_messages, polar_figure, stf_figure, mc_table_records, mc_table_column_defs, column_size_mode, degree_used, df_out_data_return)
            - glider_name (str): Displayed glider name used for outputs.
            - polar_messages (str): Informational or status messages produced by the polar model.
            - polar_figure (plotly.graph_objs.Figure): Figure containing the polar data and fit (and optional debug traces).
            - stf_figure (plotly.graph_objs.Figure): Figure containing Speed‑to‑Fly vs MacCready (and optional saved comparison traces).
            - mc_table_records (list[dict]): MacCready table rows suitable for AG Grid (fields: MC, STF, Vavg, L/D) with values converted to display units.
            - mc_table_column_defs (list[dict]): AG Grid column definition objects for the MacCready table.
            - column_size_mode (str): Column sizing mode for AG Grid (e.g., "sizeToFit").
            - degree_used (int): Effective polynomial degree applied (always >= 2).
            - df_out_data_return (dict|None): Updated serialized saved STF data for localStorage, or None if no saved data.
    """
    # Load df_out from store or initialize to None
    # Rehydrate PintArray columns with units
    if df_out_data:
        # Check if data has MultiIndex structure
        if "columns" in df_out_data and "data" in df_out_data:
            # MultiIndex format: columns is list of tuples, data is dict of lists
            df_out = pd.DataFrame(df_out_data["data"])
            # Reconstruct MultiIndex columns
            df_out.columns = pd.MultiIndex.from_tuples(df_out_data["columns"])
            # Reattach units to all data columns
            for col in df_out.columns:
                if (
                    col[0] != "MC"
                ):  # Don't reattach units to MC column (already has them)
                    df_out[col] = PA_(df_out[col], ureg.mps)
        else:
            # Legacy format: flat column structure
            df_out = pd.DataFrame(df_out_data)
            # Reattach units: all columns (MC and STF data) are in m/s
            for col in df_out.columns:
                df_out[col] = PA_(df_out[col], ureg.mps)
    else:
        df_out = None

    logger.debug(f"{dash.ctx.triggered_id=}")
    # reset accumulated data on request
    if dash.ctx.triggered_id == "clear-comparison-button":
        logger.debug("Clearing comparison data and re-enabling units change")
        df_out = None

    # Disable units change to avoid confusion
    if dash.ctx.triggered_id == "save-comparison-button":
        logger.debug("Disabling units change after saving comparison")

    logger.info(f"data from store: {data}")

    # Use default glider if none selected
    glider_name = glider_name if glider_name else DEFAULT_GLIDER_NAME
    current_glider_info = df_glider_info[df_glider_info["name"] == glider_name]

    # Just return if no glider selected
    if current_glider_info.empty:
        logger.error("update_graph: glider data not found")
        raise dash.exceptions.PreventUpdate

    # Get or create cached Glider instance to avoid duplicate CSV parsing
    current_glider = get_cached_glider(glider_name, current_glider_info)

    # Fetch stored values or initialize to None
    if data is None:
        logger.error("update_graph: no data in working-data-store")
        raise dash.exceptions.PreventUpdate

    pilot_weight = data.get("pilot_weight") if data else None
    wing_loading = data.get("wing_loading") if data else None
    v_air_horiz = data.get("v_air_horiz") if data else None
    v_air_vert = data.get("v_air_vert") if data else None

    logger.info(
        f"update_graph, from working-data-store: pilot_weight: {pilot_weight}, wing_loading: {wing_loading}, v_air_horiz: {v_air_horiz}, v_air_vert: {v_air_vert}"
    )
    logger.info(
        f"update_graph input parameters: {degree=}, {glider_name=}, {maccready=}, {goal_function=}, {show_debug_graphs=}, {write_excel_file=}\n{save_comparison=}, {clear_comparison=}, {subtract_compare=}, {units=}, {weight_or_loading=} "
    )

    # Setup units
    selected_units = UNIT_CHOICES[units]
    sink_units = selected_units["Sink"]
    speed_units = selected_units["Speed"]
    weight_units = selected_units["Weight"]
    pressure_units = selected_units["Pressure"]

    if degree is None:
        degree = DEFAULT_POLYNOMIAL_DEGREE

    if maccready is None:
        maccready = 0.0

    # Set the units for unitless items from the data store
    maccready = (maccready * sink_units).to("m/s")

    # A linear model has no solutions for the MacCready equations,
    # so the model must be a quadratic or higher order polynomial.
    degree = max(degree, 2)

    # Get a polynomial fit to the polar curve data
    current_polar: polar_calc.Polar = load_polar(
        current_glider, degree, goal_function, v_air_horiz, v_air_vert, pilot_weight
    )
    weight_factor = current_polar.get_weight_factor()

    # This label is used in legends for both the polar graph and the STF graph
    graph_trace_label = (
        (f"{current_polar.get_weight_fly().to(weight_units):.1f~P}")
        if weight_or_loading == "Pilot Weight"
        else (f"{(wing_loading * ureg('kg/m**2')).to(pressure_units):.1f~P}")
    )

    subtract_active = subtract_compare == "Subtracted" and df_out is not None

    ##################################################################
    # Graph Speed-to-Fly vs. MC setting
    # MacCready values for table, zero to 10 knots, but must be converted to m/s
    mc_graph_values = (np.arange(start=0.0, stop=10.01, step=0.02) * ureg.knots).to(
        "m/s"
    )
    df_mc_graph = current_polar.MacCready(mc_graph_values)

    mc_graph_values_converted = df_mc_graph["MC"].pint.to(sink_units).pint.magnitude
    stf_graph_values = df_mc_graph["STF"].pint.to(speed_units).pint.magnitude

    stf_graph = make_subplots(specs=[[{"secondary_y": True}]])

    trace_weight_adjusted = go.Scatter(
        x=mc_graph_values_converted,
        y=stf_graph_values,
        name="Speed-to-Fly",
        mode="lines",
        visible="legendonly" if subtract_active else True,
    )
    stf_graph.add_trace(
        trace_weight_adjusted,
        secondary_y=False,
    )

    ###################################################################
    # Plot average speed vs. MC setting
    # Clip Vavg to stay on the STF graph without expanding the Y axis too much
    plot_max = max(df_mc_graph["STF"].pint.to(speed_units).pint.magnitude)
    y = df_mc_graph["Vavg"].pint.to(speed_units).pint.magnitude.copy()
    y[(y <= -50.0) | (y >= plot_max)] = np.nan
    trace_stf = go.Scatter(
        x=df_mc_graph["MC"].pint.to(sink_units).pint.magnitude,
        y=y,
        name="Average Speed",
        mode="lines",
        visible="legendonly" if subtract_active else True,
    )
    stf_graph.add_trace(
        trace_stf,
        secondary_y=False,
    )

    ###################################################################
    # Add the saved results to the graph
    if df_out is not None:
        # Get unique column names (first level of MultiIndex)
        if isinstance(df_out.columns, pd.MultiIndex):
            # MultiIndex format
            config_names = df_out.columns.get_level_values(0).unique()
            for config_name in config_names:
                # Get MC values for this configuration
                mc_plot = df_out[(config_name, "MC")].pint.to(sink_units).pint.magnitude
                # Get the selected metric (STF or Vavg) for comparison
                metric_compare = (
                    df_out[(config_name, compare_metric)]
                    .pint.to(speed_units)
                    .pint.magnitude
                )

                # Compute the reference values to subtract (if in subtracted mode)
                if compare_metric == "STF":
                    reference_values = stf_graph_values
                else:  # Vavg
                    reference_values = (
                        df_mc_graph["Vavg"].pint.to(speed_units).pint.magnitude
                    )

                trace_saved = go.Scatter(
                    x=mc_plot,
                    y=(
                        metric_compare - reference_values
                        if subtract_active
                        else metric_compare
                    ),
                    name=f"{config_name}",
                    mode="lines",
                    line=dict(dash="dot"),
                )
                stf_graph.add_trace(
                    trace_saved,
                    secondary_y=False,
                )
        else:
            # Legacy format: flat columns (for backward compatibility)
            mc_plot = df_out["MC"].pint.to(sink_units).pint.magnitude
            for column in df_out.columns:
                if column == "MC":
                    continue
                stf_compare = df_out[column].pint.to(speed_units).pint.magnitude
                trace_saved_stf = go.Scatter(
                    x=mc_plot,
                    y=(
                        stf_compare - stf_graph_values
                        if subtract_active
                        else stf_compare
                    ),
                    name=f"{column}",
                    mode="lines",
                    line=dict(dash="dot"),
                )
                stf_graph.add_trace(
                    trace_saved_stf,
                    secondary_y=False,
                )

    ###################################################################
    # Collect results when requested
    if dash.ctx.triggered_id == "save-comparison-button":
        column_name = f"{glider_name} {graph_trace_label}"  # Degree {degree}"

        if df_out is None:
            # Create new DataFrame with MultiIndex columns
            # First level: column name (e.g., "ASW 28 100.0 kg")
            # Second level: metric type ("MC", "STF", "Vavg")
            df_out = pd.DataFrame(
                {
                    (column_name, "MC"): df_mc_graph["MC"],
                    (column_name, "STF"): df_mc_graph["STF"],
                    (column_name, "Vavg"): df_mc_graph["Vavg"],
                }
            )
            logger.debug("created df_out with MultiIndex")
        else:
            # Add new columns to existing DataFrame
            df_out[(column_name, "MC")] = df_mc_graph["MC"]
            df_out[(column_name, "STF")] = df_mc_graph["STF"]
            df_out[(column_name, "Vavg")] = df_mc_graph["Vavg"]

        logger.debug(df_out.columns)

        if write_excel_file and not _production_mode and df_out is not None:
            logger.info("Writing STF results to Excel file")
            # Save results externally
            # Sanitize glider name to avoid filesystem errors with invalid characters
            sanitized_glider_name = sanitize_filename(glider_name)
            excel_outfile_name = f"{sanitized_glider_name} stf.xlsx"
            df_out.to_excel(excel_outfile_name, sheet_name="STF", index=False)

            logger.info(f'File "{excel_outfile_name}" created successfully')

    ###################################################################
    # Plot debug graphs if requested
    if show_debug_graphs:
        trace_goal_result = go.Scatter(
            x=df_mc_graph["MC"].pint.to(sink_units).pint.magnitude,
            y=df_mc_graph["solverResult"],
            name="Solver Result",
            mode="lines",
        )
        stf_graph.add_trace(
            trace_goal_result,
            secondary_y=True,
        )

    stf_graph.update_layout(
        xaxis_title=f"MacCready Setting ({sink_units.units:~P})",
        yaxis_title=f"Speed ({speed_units.units:~P})",
        title={
            "text": "MacCready Speed-to-Fly",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis2=dict(
            title="Solver Result",  # Title for the second (right) vertical axis
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )
    ###################################################################
    # Calculate MacCready table data
    if sink_units == ureg("m/s"):
        # MacCready values for table, in m/s
        mc_table_values = np.arange(start=0.0, stop=5.1, step=0.5) * ureg.mps
    else:
        # MacCready values for table in knots, but must be converted to m/s
        mc_table_values = (np.arange(start=0.0, stop=10.1, step=1.0) * ureg.knots).to(
            ureg.mps
        )

    df_mc_table = current_polar.MacCready(mc_table_values)

    df_mc_table["MC"] = df_mc_table["MC"].pint.to(sink_units).pint.magnitude
    df_mc_table["STF"] = df_mc_table["STF"].pint.to(speed_units).pint.magnitude
    df_mc_table["Vavg"] = df_mc_table["Vavg"].pint.to(speed_units).pint.magnitude

    logger.info(current_polar.messages())

    # Store MacCready results in DataFrame for AG Grid
    # AG Grid is arranged as MC, STF, Vavg, L/D
    new_column_defs = [
        {
            "field": "MC",
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
            "headerName": f"MC ({sink_units.units:~#})",
        },
        {
            "field": "STF",
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
            "headerName": f"STF ({speed_units.units:~P})",
        },
        {
            "field": "Vavg",
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
            "headerName": f"Vavg ({speed_units.units:~P})",
        },
        {
            "field": "L/D",
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
            "headerName": "L/D",
        },
        #        "columnSize": "autoSize",
    ]

    # Graph the polar data
    polar_graph = make_subplots(specs=[[{"secondary_y": True}]])
    trace_data = go.Scatter(
        x=current_glider.get_speed_data().to(speed_units).magnitude,
        y=current_glider.get_sink_data().to(sink_units).magnitude,
        mode="markers",
        name="Polar Data",
    )
    polar_graph.add_trace(trace_data)

    # Graph the fit to the data on the same graph
    # Evaluate the polynomial for new points
    # Expand the speed range to show extrapolation used in STF calculation, if any.
    polar_speed_range = current_polar.speed_range
    stf_speed_range = (
        min(df_mc_graph["STF"]).magnitude,
        max(df_mc_graph["STF"]).magnitude,
    )
    graph_range = (
        min(polar_speed_range[0], stf_speed_range[0]),
        max(polar_speed_range[1], stf_speed_range[1]),
    )

    # Make the speed points at 0.5 m/s intervals
    speed_mps_magnitude = np.arange(
        round((2 * graph_range[0]) / 2.0) - 1, 1 + round(2 * graph_range[1]) / 2.0, 0.5
    )

    speed = (speed_mps_magnitude * ureg("m/s")).to(speed_units).magnitude
    sink = current_polar.sink(
        speed_mps_magnitude, weight_correction=False, include_airmass=False
    )
    sink = (sink * ureg("m/s")).to(sink_units).magnitude

    trace_fit = go.Scatter(x=speed, y=sink, name=f"Fit, degree={degree}")
    polar_graph.add_trace(trace_fit)

    if show_debug_graphs:
        goal_function_values = current_polar.goal_function(
            speed_mps_magnitude, maccready.magnitude
        )
        # goal_function_values[
        #     (goal_function_values >= 10) | (goal_function_values <= -10)
        # ] = np.nan
        trace_goal = go.Scatter(
            x=speed,
            y=goal_function_values,
            name="Goal Function",
        )
        polar_graph.add_trace(trace_goal)

    if show_debug_graphs:
        # Graph the residuals (difference between the data and the fit)
        speed_data = current_glider.get_speed_data().to(speed_units)
        sink_fit = current_polar.sink(
            current_glider.get_speed_data().magnitude,
            weight_correction=False,
            include_airmass=False,
        )
        residual = (current_glider.get_sink_data() - (sink_fit * ureg("m/s"))).to(
            sink_units
        )

        trace_residuals = go.Scatter(
            x=speed_data.magnitude, y=residual.magnitude, name="Residuals"
        )
        polar_graph.add_trace(trace_residuals, secondary_y=True)

    # Add the weight-adjusted polar, but only if the all-up weight differs from the reference weight
    if weight_factor != 1.0 or v_air_vert != 0.0:
        # sink = (current_polar.sink(speed_mps_magnitude, weight_correction=True, include_airmass=False)*ureg('m/s')).to(sink_units).magnitude
        trace_weight_adjusted = go.Scatter(
            x=speed * weight_factor,
            y=sink * weight_factor,
            name=graph_trace_label,
        )
        polar_graph.add_trace(trace_weight_adjusted)
    """
    # Alternate adjustment: Add the weight-adjusted polar, but only if the all-up weight differs from the reference weight
    if (weight_factor != 1.0 or v_air_vert != 0.0):
        sink = (current_polar.sink(speed_mps_magnitude, weight_correction=True, include_airmass=False)*ureg('m/s')).to(sink_units).magnitude
        trace_weight_adjusted = go.Scatter(
            x=speed,
            y=sink,
            name=f"Alternate adjustment")
        polar_graph.add_trace(trace_weight_adjusted)
    """
    polar_graph.update_layout(
        xaxis_title=f"Speed ({speed_units.units:~P})",
        yaxis_title=f"Sink ({sink_units.units:~P})",
        title={
            "text": f"{glider_name} Polar",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis2=dict(
            title="Residuals",  # Title for the second (right) axis
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )

    polar_graph.update_yaxes(tickformat=".1f", secondary_y=False)

    # Convert df_out to dict for storage, or None if df_out is None
    # Strip pint units (keep only magnitudes) for localStorage serialization
    if df_out is not None:
        if isinstance(df_out.columns, pd.MultiIndex):
            # Serialize MultiIndex structure
            df_out_data_return = {
                "columns": [tuple(col) for col in df_out.columns.tolist()],
                "data": {
                    str(col): (
                        df_out[col].pint.magnitude.tolist()
                        if hasattr(df_out[col], "pint")
                        else df_out[col].tolist()
                    )
                    for col in df_out.columns
                },
            }
        else:
            # Legacy flat format
            df_out_data_return = {
                col: (
                    df_out[col].pint.magnitude.tolist()
                    if hasattr(df_out[col], "pint")
                    else df_out[col].tolist()
                )
                for col in df_out.columns
            }
    else:
        df_out_data_return = None

    logger.debug("update_graph return\n")
    return (
        glider_name,
        current_polar.messages(),
        polar_graph,
        stf_graph,
        df_mc_table.to_dict("records"),
        new_column_defs,
        "sizeToFit",
        degree,
        (
            df_out_data_return is None
        ),  # disable the "Clear Comparison" button if there is no data saved
        df_out_data_return,
    )


##################################################################
# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")

    # Determine production mode from environment variable, with fallback to port-based detection
    env_production = os.environ.get("PRODUCTION_MODE", "").lower()
    env_environment = os.environ.get("ENVIRONMENT", "").lower()

    if env_production:
        _production_mode = env_production in ("true", "1", "yes")
    elif env_environment:
        _production_mode = env_environment == "production"
    else:
        # Fallback to port-based detection for compatibility
        _production_mode = port != 8050

    if not _production_mode:
        logger.info(
            f"Starting development server at http://localhost:{port}, debug={debug}, production_mode={_production_mode}"
        )
        app.run(debug=debug)
    else:
        logger.info(
            f"Starting production server on port {port}, debug={debug}, production_mode={_production_mode}"
        )
        app.run(host="0.0.0.0", port=port, debug=False)
