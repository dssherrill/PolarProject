#!/usr/bin/env python3
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash_ag_grid as dag

import polar_calc

import pint
import pint_pandas

import logging

# Get access to the one-and-only UnitsRegistry instance
from units import ureg
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

# define global variable
pilot_weight_kg = None
df_out = None

# Read list of available polars from a file
df_glider_info = pd.read_json('datafiles/gliderInfo.json')
current_glider = df_glider_info[df_glider_info['name'] == 'ASK 21']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# polynomical fit degree (order)
DEFAULT_POLYNOMIAL_DEGREE = 5
weight_factor = 1.0

# display options
METRIC_UNITS = {'Speed': ureg('kph'), 'Sink': ureg('m/s'), 'Weight': ureg('kg')}
US_UNITS = {'Speed': ureg('knots'), 'Sink': ureg('knots'), 'Weight': ureg('lbs')}
UNIT_CHOICES = {'Metric' : METRIC_UNITS, 'US' : US_UNITS}

def load_polar(current_glider, degree, goal_function, v_air_horiz, v_air_vert, pilot_weight):
    current_polar = polar_calc.Polar(current_glider, degree, goal_function, v_air_horiz, v_air_vert, pilot_weight)
    speed, _ = current_polar.get_polar()

    # Evaluate the polynomial for new points
    speed = np.linspace(min(speed).magnitude, max(speed).magnitude, 100)
    dfFit = pd.DataFrame({'Speed': PA_(speed, ureg.mps), 'Sink': PA_(current_polar.Sink(speed), ureg.mps)})
    
    return dfFit, current_polar

# dummy data to setup the AG Grid as a MacCready table
initial_data = pd.DataFrame({'MC': [0], 'STF': [0], 'Vavg': [0], 'L/D': [0]})

# Format the data for the dcc.Dropdown 'options' property
# It requires a list of dictionaries with 'label' (what the user sees)
# and 'value' (the actual data used internally) keys.
dropdown_options = [
    {'label': row['name'], 'value': row['name']}
    for index, row in df_glider_info.iterrows()
]

# Styles for Flexbox container and items
container_style = {
    "display": "flex",
    "flex-wrap": "wrap", # Allows items to stack on small screens
    # "gap": "20px",       # Space between items
    # "padding": "20px"
}

# Responsive item: occupies half width on large screens, full width on small
item_style = {
    "flex": "0 0 450px", # grow=1, shrink=1, base_width=450px
    "min-width": "300px" 
}

graph_style = {
    "flex": "1 1 800px", # grow=1, shrink=1, base_width=450px
    "min-width": "600px",
    "min-height": "400px"
}

full_width_class = "d-flex w-100 justify-content-center bg-light p-3"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div(id='main-title', className="text-primary fs-3",  )
        ], width=12)
    ], className=full_width_class),
    dbc.Row([
        # Input controls group
        dbc.Col([
            dbc.Row([
                html.Label("Select a glider:"),
                dcc.Dropdown(
                    id="glider-dropdown",
                    options=dropdown_options,
                    value='ASW 28', # dropdown_options[0]['value'] if dropdown_options else None, # Set a default value if options exist
                    placeholder="Select a glider...",
                ),
                html.Div(id="output-container"),
                            
            ], className="mb-3"),

            

            dbc.Row([
                dbc.Row([
                #--- reference weight
                    dbc.Label("Reference weight (kg):", html_for="ref-weight-input", id="ref-weight-label", ),
                    dcc.Input(id="ref-weight-input", type="number", placeholder="400", readOnly=True),

                #--- empty weight
                    dbc.Label("Empty weight (kg):", html_for="empty-weight-input", id="empty-weight-label"),
                    dcc.Input(id="empty-weight-input", type="number", placeholder="300", readOnly=True),

                #--- pilot weight
                    dbc.Label("Pilot + Ballast weight (kg):", html_for="pilot-weight-input", id="pilot-weight-label"),
                    dcc.Input(id="pilot-weight-input", type="number", placeholder="100", ), # , debounce=True),
                    ]),
                ], className="mb-3"),

            dbc.Row([
                dbc.RadioItems(options=['Metric', 'US'],
                            value='Metric',
                            inline=False,
                            id='radio-units',
                            )
            ], className="mb-3"),

            # dbc.Row([
            #     dbc.Col(
            #         dbc.Button("Submit", id="submit-button", color="primary", className="mt-3"),
            #         # width={"size": 6, "offset": 3}
            #     )
            # ], className="mb-3"),
        ], ), # width=2),
        dbc.Col ([
            #--- Polynomial degree
            dbc.Row([
                    dbc.Label("Polynomial degree:", html_for="poly-degree"),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Input(id="poly-degree", type="number", placeholder=DEFAULT_POLYNOMIAL_DEGREE, min=2), # style={"width": "1"}),
                    ]), # width=10),
            ]),

            # Statistics output group
            dbc.Row([
                html.Br(),
                dcc.Markdown("Statistics", 
                             style={"whiteSpace": "pre-line", "fontFamily": "Courier, serif"}, 
                             id="statistics",
                             mathjax=True,  # Enable LaTeX rendering
                             dangerously_allow_html=True ), # enable html display without purify
            ], 
            #width={"size": 2, "offset": 1},
            className="mt-3 mb-3"),
            ],), # width=2),
    ]),

    # dbc.Row([
    #     dbc.Col(
    #         html.Div([dcc.Slider(300, 600, 1, marks={i: f'{i}' for i in range(300, 610, 50)}, 
    #                              value=320, 
    #                              tooltip={
    #                                  "always_visible": True,
    #                                 "template": "{value}"
    #                                 },
    #                             id='weight-slider'),
    #                             html.Div(id='slider-output-container')
    #               ]),
    #               width = 4
    #     ),

    #     dbc.Col(
    #         html.Div()
    #     )
    # ]),

    #--- Graphs
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='graph-polar', style=graph_style)
        ], ), # width=4),

        dbc.Col([
            dcc.Graph(figure={}, id='graph-stf', style=graph_style)
        ], ), # width=4),
    ]),
    dbc.Row([
    dag.AgGrid(rowData=initial_data.to_dict('records'),
                columnDefs=[{"field": i, 'valueFormatter': {"function": "d3.format('.1f')(params.value)"},
                            'type': 'numericColumn'
                            } for i in initial_data.columns],
                            columnSize="autoSize",
                            id='mcAgGrid')
    ], style=item_style), # width=3)

    dbc.Row([
        html.Div('Airmass Movement', className="text-primary fs-3"),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Label("Horizontal speed", id='horizontal-speed-label'),
                    dcc.Input(id="airmass-horizontal-speed", type="number", placeholder="0", step="5.0"),
                ], ), # width=2),
                dbc.Row([
                    dbc.Label("Vertical speed", id='vertical-speed-label'),
                    dcc.Input(id="airmass-vertical-speed", type="number", placeholder="0", min="-10", max="10", step="0.25"),
                ], ), # width=2),
            ], width=3),
        ])
    ], className=full_width_class),
    dbc.Row([
        html.Div('Debug Options', className="text-primary fs-3 mt-5"),
        dbc.Switch(
            id="toggle-switch-debug",
            label="Include debug graphs",
            value=False,  # Initial state
            className="ms-3"
        ),
    # ]),

    # dbc.Row([
        dbc.Switch(
            id="toggle-switch-dump",
            label="Dump file",
            value=False,  # Initial state
            className="ms-3"
        ),
        
        dbc.Col([
            dbc.Row([
            html.Div('Solution', className="text-primary fs-3 mt-5"),
                dbc.RadioItems(options=['Reichmann', 'Test'],
                               value='Reichmann',
                               inline=False,
                               id='radio-goal',
                               )
            ], className="mb-3"),

            dbc.Label("MacCready Value for Goal Function graph",html_for="ref-weight-input", className="w-2"),
            html.Br(),
            dcc.Input(id="maccready-input", type="number", placeholder="0"),
        ], ), # width=2),
    ], style={"flex": "1 1 100%", }), # "margin-top": "20px"}),
], style=container_style, fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='main-title', component_property='children'),
    Output(component_id='horizontal-speed-label', component_property='children'),
    Output(component_id='vertical-speed-label', component_property='children'),
    Output(component_id='statistics', component_property='children'),
    Output(component_id='ref-weight-input', component_property='placeholder'),
    Output(component_id='empty-weight-input', component_property='placeholder'),
    Output(component_id='pilot-weight-input', component_property='placeholder'),
    Output(component_id='pilot-weight-input', component_property='value'),
    Output(component_id='graph-polar', component_property='figure'),
    Output(component_id='graph-stf', component_property='figure'),
    Output(component_id='mcAgGrid', component_property='rowData'),
    Output(component_id='mcAgGrid', component_property='columnDefs'),
    Output(component_id='mcAgGrid', component_property='columnSize'),
    Output(component_id='poly-degree', component_property='value'),
    Output(component_id='ref-weight-label', component_property='children'),
    Output(component_id='empty-weight-label', component_property='children'),
    Output(component_id='pilot-weight-label', component_property='children'),
    

    # Input('pilot-weight-input', 'n-submit'),
    # Input('submit-button', 'n_clicks'),
    Input(component_id='poly-degree', component_property='value'),
    Input(component_id='glider-dropdown', component_property='value'),
    Input(component_id='radio-units', component_property='value'),
    Input(component_id='maccready-input', component_property='value'),
    Input(component_id='pilot-weight-input', component_property='value'),
    Input(component_id='radio-goal', component_property='value'),
    Input(component_id='airmass-horizontal-speed', component_property='value'),
    Input(component_id='airmass-vertical-speed', component_property='value'),    
    Input(component_id='toggle-switch-debug', component_property='value'),
    Input(component_id='toggle-switch-dump', component_property='value'),
    State(component_id='ref-weight-input', component_property='value'),
    State(component_id='empty-weight-input', component_property='value'),
#    prevent_initial_call=True
)
def update_graph(degree, glider_name, units, maccready, pilot_weight, goal_function, v_air_horiz, v_air_vert, show_debug_graphs, write_excel_file, reference_weight, empty_weight):
    current_glider = df_glider_info[df_glider_info['name'] == glider_name]

    global pilot_weight_kg
    global df_out
    # handle display units option
    selected_units = UNIT_CHOICES[units]
    sink_units = selected_units['Sink']
    speed_units = selected_units['Speed']
    weight_units = selected_units['Weight']

    if degree is None:
        degree = DEFAULT_POLYNOMIAL_DEGREE

    if maccready is None:
        maccready = 0.0

    if v_air_horiz is None:
        v_air_horiz = 0.0

    if v_air_vert is None:
        v_air_vert = 0.0

    # capture the pilot weight each time it changes
    if dash.ctx.triggered_id == 'pilot-weight-input':
        logger.debug('pilot-weight-input clicked')
        if pilot_weight is None:
            pilot_weight_kg = None
        else:
            pilot_weight_kg = (pilot_weight * weight_units).to('kg').magnitude

    # Set the units for unitless items from the user interface
    maccready = maccready * sink_units.to('m/s')
    v_air_horiz = v_air_horiz * speed_units.to('m/s')
    v_air_vert = v_air_vert * speed_units.to('m/s')

    # A linear model has no solutions for the MacCready equations, 
    # so the model must be a quadratic or higher order polynomial.
    degree = max(degree, 2)

    # Get a polynomial fit to the polar curve data
    df_fit, current_polar = load_polar(current_glider, degree, goal_function, v_air_horiz, v_air_vert, pilot_weight_kg)
    weight_factor = current_polar.get_weight_factor()

    # Graph the polar data
    polar_graph = make_subplots(specs=[[{"secondary_y": True}]])
    trace_data = go.Scatter(x=current_polar.getSpeedData().to(speed_units).magnitude,
                        y=current_polar.getSinkData().to(sink_units).magnitude,
                        mode='markers',
                        name='Polar Data',)
    polar_graph.add_trace(trace_data)

    # Graph the fit to the data on the same graph
    trace_fit = go.Scatter(x=df_fit['Speed'].pint.to(speed_units).pint.magnitude,
                        y=df_fit['Sink'].pint.to(sink_units).pint.magnitude,
                        name=f"Fit, degree={degree}")
    polar_graph.add_trace(trace_fit)

    if show_debug_graphs:
        # Graph the residuals (difference between the data and the fit)
        speed_data = current_polar.getSpeedData().to(speed_units)
        sink_fit = current_polar.Sink(current_polar.getSpeedData().magnitude)
        resid = current_polar.getSinkData().to(sink_units) - (sink_fit * ureg('m/s')).to(sink_units)
        trace_residuals = go.Scatter(x=speed_data.magnitude,
                            y=resid.magnitude,
                            name=f"Residuals")
        polar_graph.add_trace(trace_residuals, secondary_y=True)

    # Add the weight-adjusted polar, but only if the all-up weight differs from the reference weight
    if (weight_factor != 1.0):
        trace_weight_adjusted = go.Scatter(x=df_fit['Speed'].pint.to(speed_units).pint.magnitude * weight_factor,
                            y=df_fit['Sink'].pint.to(sink_units).pint.magnitude * weight_factor,
                            name=f"Adjusted to {(current_polar.get_weight_fly() * ureg('kg')).to(weight_units).magnitude:.1f} {weight_units.units:~P}")
        polar_graph.add_trace(trace_weight_adjusted)

    polar_graph.update_layout(
        xaxis_title=f"Speed ({speed_units.units:~P})",
        yaxis_title=f"Sink ({sink_units.units:~P})",
        title={
        'text': f"{glider_name} Polar",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        },
        yaxis2=dict(
            title="Residuals", # Title for the second (right) axis
            overlaying="y",
            side="right",),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.3,
            xanchor="center", 
            x=0.5)
        )

    polar_graph.update_yaxes(tickformat=".1f", secondary_y=False) 

    # Graph Speed-to-Fly vs. MC setting
    # MacCready values for table, zero to 6 knots, but must be coverted to m/s
    mc_table =  (np.arange(start=0.0, stop=6.1, step=0.02) * ureg.knots).to('m/s')
    df_mc = current_polar.MacCready(mc_table)
    v =df_fit['Speed'].pint.magnitude

    if show_debug_graphs:
        goal_function_values = current_polar.goal_function(v, maccready.magnitude)
        goal_function_values[(goal_function_values >= 10) | (goal_function_values <= -10)] = np.nan
        trace_goal = go.Scatter(x=df_fit['Speed'].pint.to(speed_units).pint.magnitude,
                            y=goal_function_values,
                            name='Goal Function',)
        polar_graph.add_trace(trace_goal)

    stf_graph = make_subplots(specs=[[{"secondary_y": True}]])
    trace_weight_adjusted = go.Scatter(x = df_mc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = df_mc['STF'].pint.to(speed_units).pint.magnitude, 
                        name='Speed-to-Fly',
                        mode='lines')
    stf_graph.add_trace(trace_weight_adjusted, secondary_y=False,)

    # Collect results if Excel output requested
    if write_excel_file:
        if (df_out is None):
            df_out = pd.DataFrame(df_mc['MC'].pint.to(sink_units).pint.magnitude)
            logger.debug('created df_out')
        column_name =  f'Degree {degree}'
        if column_name in df_out.columns:
            df_out[column_name] = df_mc['STF'].pint.to(speed_units).pint.magnitude
        else:
            df_out = pd.concat([df_out, df_mc['STF'].pint.to(speed_units).pint.magnitude], axis=1)
            df_out.rename(columns={'STF': column_name}, inplace=True)

        logger.debug(df_out.columns)
        # df_out[f'degree {degree}'] = df_mc['STF'].pint.to(speed_units).pint.magnitude

        # Save results externally
        # Open the file in write mode ('w') with newline=''
        excel_outfile_name = f'{glider_name} stf.xlsx'
        df_out.to_excel(excel_outfile_name, sheet_name='STF', index=False)

        logger.info(f'File "{excel_outfile_name}" created successfully')
    else:
        # Delete any accumulated data
        df_out = None

    plot_max = max(df_mc['STF'].pint.to(speed_units).pint.magnitude)
    y = df_mc['Vavg'].pint.to(speed_units).pint.magnitude
    y[(y <= -50.0) | (y >= plot_max)] = np.nan
    trace_stf = go.Scatter(x = df_mc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = y,
                        name='Average Speed',
                        mode='lines')
    stf_graph.add_trace(trace_stf, secondary_y=False,)

    if show_debug_graphs:
        trace_goal_result = go.Scatter(x = df_mc['MC'].pint.to(sink_units).pint.magnitude, 
                            y = df_mc['solverResult'],
                            name='Solver Result',
                            mode='lines')
        stf_graph.add_trace(trace_goal_result, secondary_y=True,)

    stf_graph.update_layout(
        xaxis_title=f"MacCready Setting ({sink_units.units:~P})",
        yaxis_title=f"Speed ({speed_units.units:~P})",
        title={
        'text': 'MacCready Speed-to-Fly',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        yaxis2=dict(
        title="Solver Result", # Title for the second (right) axis
        overlaying="y",
        side="right",),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.3,
            xanchor="center", 
            x=0.5)
        )

    if (sink_units == ureg('m/s')):
        # MacCready values for table, in m/s
        mc_table =  np.arange(start=0.0, stop=3.1, step=0.5) * ureg.mps
    else:
        # MacCready values for table in knots, but must be coverted to m/s
        mc_table =  (np.arange(start=0.0, stop=6.1, step=1.0) * ureg.knots).to(ureg.mps)
    
    df_mc = current_polar.MacCready(mc_table)

    df_mc['MC'] = df_mc['MC'].pint.to(sink_units).pint.magnitude
    df_mc['STF'] = df_mc['STF'].pint.to(speed_units).pint.magnitude
    df_mc['Vavg'] = df_mc['Vavg'].pint.to(speed_units).pint.magnitude

    # These are all in kg but stored as floats without units    
    reference_weight = current_glider['referenceWeight'].iloc[0] * ureg('kg')
    empty_weight = current_glider['emptyWeight'].iloc[0] * ureg('kg')
    reference_pilot_weight = reference_weight - empty_weight 

    if pilot_weight_kg is None:
        pilot_weight_out = None
    else:
        pilot_weight_out = f"{(pilot_weight_kg * ureg('kg')).to(selected_units['Weight']).magnitude:.1f}"

    logger.info(current_polar.messages())

    # pd.DataFrame({'MC': [0], 'STF': [0], 'Vavg': [0], 'L/D': [0]})
    new_column_defs = [
        {"field": "MC", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.1f')(params.value)"}, "headerName": f"MC ({sink_units.units:~#})"},
        {"field": "STF", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.1f')(params.value)"}, "headerName": f"STF ({speed_units.units:~P})"},
        {"field": "Vavg", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.1f')(params.value)"}, "headerName": f"Vavg ({speed_units.units:~P})"},
        {"field": "L/D", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.1f')(params.value)"}, "headerName": "L/D"},
#        "columnSize": "autoSize",
    ]

    logger.debug('update_graph return\n')
    return (glider_name, 
            f"Horizontal speed ({speed_units.units:~P})",
            f"Vertical speed ({speed_units.units:~P})",
            current_polar.messages(), 
            f"{reference_weight.to(selected_units['Weight']).magnitude:.1f}",
            f"{empty_weight.to(selected_units['Weight']).magnitude:.1f}",
            f"{reference_pilot_weight.to(selected_units['Weight']).magnitude:.1f}",
            pilot_weight_out,
            polar_graph, 
            stf_graph, 
            df_mc.to_dict('records'), 
            new_column_defs,
            "sizeToFit",
            degree,
            f"Reference weight ({weight_units.units:~P}):",
            f"Empty weight ({weight_units.units:~P}):",
            f"Pilot + Ballast weight ({weight_units.units:~P}):",
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)