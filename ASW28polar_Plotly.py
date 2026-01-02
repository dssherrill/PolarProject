#!C:\Users\david\AppData\Local\Programs\Python\Python312\python.exe
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash_ag_grid as dag

import polar

import dash_bootstrap_components as dbc

import pint
import pint_pandas

# Get access to the one-and-only UnitsRegistry instance
from units import ureg
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity

# define global variable
pilot_weight_kg = None

# Read list of available polars from a file
df_glider_info = pd.read_json('datafiles/gliderInfo.json')
current_glider = df_glider_info[df_glider_info['name'] == 'ASK 21']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# polynomical fit degree (order)
DEFAULT_POLYNOMIAL_DEGREE = 5
weight_factor = 1.0

# display options
METRIC_UNITS = {'Speed': ureg('kph'), 'Sink': ureg('mps'), 'Weight': ureg('kg')}
US_UNITS = {'Speed': ureg('knots'), 'Sink': ureg('knots'), 'Weight': ureg('lbs')}
UNIT_CHOICES = {'Metric' : METRIC_UNITS, 'US' : US_UNITS}

def polar_calc(current_glider, v_air_horiz, v_air_vert, pilot_weight, degree):
    current_polar = polar.Polar(current_glider, degree, v_air_horiz, v_air_vert, pilot_weight)
    speed, sink = current_polar.get_polar()

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

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div(id='main-title', className="text-primary fs-3")
    ]),
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
                            value='US',
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
        ], width=2),
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
            ], width=5),
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
            dcc.Graph(figure={}, id='graph-polar')
        ], width=5),

        dbc.Col([
            dcc.Graph(figure={}, id='graph-stf')
        ], width=4),

        dbc.Col([
            dag.AgGrid(rowData=initial_data.to_dict('records'),
                       columnDefs=[{"field": i, 'valueFormatter': {"function": "d3.format('.1f')(params.value)"},
                                    'type': 'numericColumn'
                                    } for i in initial_data.columns],
                                    columnSize="autoSize",
                                    id='mcAgGrid')
        ])
    ]),
    dbc.Row([
        html.Div('Airmass Movement', className="text-primary fs-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizontal speed"),
                dcc.Input(id="airmass-horizontal-speed", type="number", placeholder="0"),
            ], width=2),
            dbc.Col([
                dbc.Label("Vertical speed"),
                dcc.Input(id="airmass-vertical-speed", type="number", placeholder="0"),
            ], width=2),
            ]), # width=),
            ]),
    dbc.Row([
        html.Div('Debug Options', className="text-primary fs-3 mt-5"),
        dbc.Switch(
            id="toggle-switch-debug",
            label="Include debug graphs",
            value=False,  # Initial state
            className="m-3"
        ),
        dbc.Col([
            dbc.Label("MacCready Value for Goal Function graph",html_for="ref-weight-input", className="w-2"),
            html.Br(),
            dcc.Input(id="maccready-input", type="number", placeholder="0"),
        ], ), # width=2),
    ]),
], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='main-title', component_property='children'),
    Output(component_id='statistics', component_property='children'),
    Output(component_id='ref-weight-input', component_property='placeholder'),
    Output(component_id='empty-weight-input', component_property='placeholder'),
    Output(component_id='pilot-weight-input', component_property='placeholder'),
    Output(component_id='pilot-weight-input', component_property='value'),
    Output(component_id='graph-polar', component_property='figure'),
    Output(component_id='graph-stf', component_property='figure'),
    Output(component_id='mcAgGrid', component_property='rowData'),
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
    Input(component_id='airmass-horizontal-speed', component_property='value'),
    Input(component_id='airmass-vertical-speed', component_property='value'),    
    Input(component_id='toggle-switch-debug', component_property='value'),
    State(component_id='ref-weight-input', component_property='value'),
    State(component_id='empty-weight-input', component_property='value'),
#    prevent_initial_call=True
)
def update_graph(degree, glider_name, units, maccready, pilot_weight, v_air_horiz, v_air_vert, show_debug_graphs, reference_weight, empty_weight):
    current_glider = df_glider_info[df_glider_info['name'] == glider_name]

    global pilot_weight_kg

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
        print('pilot-weight-input clicked')
        if pilot_weight == None:
            pilot_weight_kg = None
        else:
            pilot_weight_kg = (pilot_weight * weight_units).to('kg').magnitude

    # Set the units for unitless items from the user interface
    maccready = maccready * sink_units.to('mps')
    v_air_horiz = v_air_horiz * speed_units.to('mps')
    v_air_vert = v_air_vert * speed_units.to('mps')

    # A linear model has no solutions for the MacCready equations, 
    # so the model must be a quadratic or higher order polynomial.
    degree = max(degree, 2)

    # Get a polynomial fit to the polar curve data
    df_fit, current_polar = polar_calc(current_glider, v_air_horiz, v_air_vert, pilot_weight_kg, degree)
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
        resid = current_polar.getSinkData().to(sink_units) - (sink_fit * ureg('mps')).to(sink_units)
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
    mc_table =  (np.arange(start=0.0, stop=6.1, step=0.02) * ureg.knots).to('mps')
    df_mc = current_polar.MacCready(mc_table)
    v =df_fit['Speed'].pint.magnitude

    if show_debug_graphs:
        goal_function_values = current_polar.goal_function(v, maccready.magnitude)
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

    trace_stf = go.Scatter(x = df_mc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = df_mc['Vavg'].pint.to(speed_units).pint.magnitude, 
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
        'text': f"MacCready Speed-to-Fly",
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

    if (sink_units == ureg('mps')):
        # MacCready values for table, in m/s
        mc_table =  np.arange(start=0.0, stop=3.1, step=0.5) * ureg.mps
    else:
        # MacCready values for table in knots, but must be coverted to m/s
        mc_table =  (np.arange(start=0.0, stop=6.1, step=1.0) * ureg.knots).to(ureg.mps)
    
    df_mc = current_polar.MacCready(mc_table)

    df_mc['MC'] = df_mc['MC'].pint.to(sink_units).pint.magnitude
    df_mc['STF'] = df_mc['STF'].pint.to(speed_units).pint.magnitude
    df_mc['Vavg'] = df_mc['Vavg'].pint.to(speed_units).pint.magnitude

    # These are all in kg but stored at floats without units    
    reference_weight = current_glider['referenceWeight'].iloc[0] * ureg('kg')
    empty_weight = current_glider['emptyWeight'].iloc[0] * ureg('kg')
    reference_pilot_weight = reference_weight - empty_weight 

    if pilot_weight_kg == None:
        pilot_weight_out = None
    else:
        pilot_weight_out = f"{(pilot_weight_kg * ureg('kg')).to(selected_units['Weight']).magnitude:.1f}"

    print(current_polar.messages())

    print('update_graph return\n')
    return (glider_name, 
            current_polar.messages(), 
            f"{reference_weight.to(selected_units['Weight']).magnitude:.1f}",
            f"{empty_weight.to(selected_units['Weight']).magnitude:.1f}",
            f"{reference_pilot_weight.to(selected_units['Weight']).magnitude:.1f}",
            pilot_weight_out,
            polar_graph, 
            stf_graph, 
            df_mc.to_dict('records'), 
            degree,
            f"Reference weight ({selected_units['Weight'].units:~P}):",
            f"Empty weight ({selected_units['Weight'].units:~P}):",
            f"Pilot + Ballast weight ({selected_units['Weight'].units:~P}):",
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)