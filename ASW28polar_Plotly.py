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

dfGliderInfo = pd.read_json('datafiles/gliderInfo.json')
currentGlider = dfGliderInfo[dfGliderInfo['name'] == 'ASK 21']

#dfGliderInfo
# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# polynomical fit degree (order)
DEFAULT_POLYNOMIAL_DEGREE = 5
weight_factor = 1.0

# display options
metricUnits = {'Speed': ureg('kph'), 'Sink': ureg('mps'), 'Weight': ureg('kg')}
UsUnits = {'Speed': ureg('knots'), 'Sink': ureg('knots'), 'Weight': ureg('lbs')}
unitLabels = {'Metric' : metricUnits, 'knots' : UsUnits}

def polarCalc(current_glider, v_air_horiz, v_air_vert, pilot_weight, degree):
    myPolar = polar.Polar(current_glider, degree, v_air_horiz, v_air_vert, pilot_weight)
    speed, sink = myPolar.get_polar()

    # Evaluate the polynomial for new points
    speed = np.linspace(min(speed).magnitude, max(speed).magnitude, 100)
    dfFit = pd.DataFrame({'Speed': PA_(speed, ureg.mps), 'Sink': PA_(myPolar.Sink(speed), ureg.mps)})
    
    return dfFit, myPolar

# dummy data to setup the AG Grid as a MacCready table
initial_data = pd.DataFrame({'MC': [0], 'STF': [0], 'Vavg': [0], 'L/D': [0]})

# Format the data for the dcc.Dropdown 'options' property
# It requires a list of dictionaries with 'label' (what the user sees)
# and 'value' (the actual data used internally) keys.
dropdown_options = [
    {'label': row['name'], 'value': row['name']}
    for index, row in dfGliderInfo.iterrows()
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
                    dbc.Label("Reference weight (kg):", html_for="ref-weight-input"),
                    dcc.Input(id="ref-weight-input", type="number", placeholder="400", readOnly=True),

                #--- empty weight
                    dbc.Label("Empty weight (kg):", html_for="empty-weight-input"),
                    dcc.Input(id="empty-weight-input", type="number", placeholder="300", readOnly=True),

                #--- reference weight
                    dbc.Label("Pilot + Ballast weight (kg):", html_for="pilot-weight-input"),
                    dcc.Input(id="pilot-weight-input", type="number", placeholder="100"), # , debounce=True),
                    ]),
                ], className="mb-3"),

            dbc.Row([
                dbc.RadioItems(options=['Metric', 'knots'],
                            value='knots',
                            inline=False,
                            id='radio-units',
                            )
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(
                    dbc.Button("Submit", id="submit-button", color="primary", className="mt-3"),
                    # width={"size": 6, "offset": 3}
                )
            ], className="mb-3"),
        ], width=3),
        dbc.Col ([
            #--- Polynomial degree
            dbc.Row([
                    dbc.Label("Polynomial degree:", html_for="poly-degree"),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Input(id="poly-degree", type="number", placeholder=DEFAULT_POLYNOMIAL_DEGREE), # style={"width": "1"}),
                    ]), # width=10),
            ]),

            # Statistics output group
            dbc.Row([
                dcc.Markdown("Statistics", 
                             style={"whiteSpace": "pre-line", "fontFamily": "Courier, serif"}, 
                             id="statistics",
                             mathjax=True,  # Enable LaTeX rendering
                             dangerously_allow_html=True ), # enable html display without purify
            ], 
            #width={"size": 2, "offset": 1},
            className="mb-3"),
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

    dbc.Row([
        #--- MacCready input
        dbc.Col([
                dbc.Label("MacCready Setting", html_for="ref-weight-input"),
                dcc.Input(id="maccready-input", type="number", placeholder="0"),
            ], width=5),
            ]),
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

], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='main-title', component_property='children'),
    Output(component_id='statistics', component_property='children'),
    Output(component_id='ref-weight-input', component_property='placeholder'),
    Output(component_id='empty-weight-input', component_property='placeholder'),
    Output(component_id='pilot-weight-input', component_property='placeholder'),
    Output(component_id='graph-polar', component_property='figure'),
    Output(component_id='graph-stf', component_property='figure'),
    Output(component_id='mcAgGrid', component_property='rowData'),
    Output(component_id='poly-degree', component_property='value'),

    #Input('pilot-weight-input', 'n-submit'),
    Input('submit-button', 'n_clicks'),
    Input(component_id='poly-degree', component_property='value'),
    Input(component_id='glider-dropdown', component_property='value'),
    Input(component_id='radio-units', component_property='value'),
    Input(component_id='maccready-input', component_property='value'),
    Input(component_id='pilot-weight-input', component_property='value'),
    Input(component_id='airmass-horizontal-speed', component_property='value'),
    Input(component_id='airmass-vertical-speed', component_property='value'),    
    State(component_id='ref-weight-input', component_property='value'),
    State(component_id='empty-weight-input', component_property='value'),
#    prevent_initial_call=True
)
def update_graph(n_clicks, degree, glider_name, units, maccready, pilot_weight, v_air_horiz, v_air_vert, reference_weight, empty_weight):
    current_glider = dfGliderInfo[dfGliderInfo['name'] == glider_name]

    # handle display units option
    selectedUnits = unitLabels[units]
    sink_units = selectedUnits['Sink']
    speed_units = selectedUnits['Speed']
    weight_units = selectedUnits['Weight']

    if degree is None:
        degree = DEFAULT_POLYNOMIAL_DEGREE

    if maccready is None:
        maccready = 0.0

    if v_air_horiz is None:
        v_air_horiz = 0.0

    if v_air_vert is None:
        v_air_vert = 0.0

    # Set the units for unitless items from the user interface
    maccready = maccready * selectedUnits['Sink'].to('mps')
    v_air_horiz = v_air_horiz * selectedUnits['Speed'].to('mps')
    v_air_vert = v_air_vert * selectedUnits['Speed'].to('mps')

    # A linear model has no solutions for the MacCready equations, 
    # so the model must be a quadratic or higher order polynomial.
    degree = max(degree, 2)

    # Get a polynomial fit to the polar curve data
    dfFit, myPolar = polarCalc(current_glider, v_air_horiz, v_air_vert, pilot_weight, degree)
    wFactor = myPolar.get_weight_factor()

    # Graph the polar data
    polarGraph = make_subplots(specs=[[{"secondary_y": True}]])
    trace1 = go.Scatter(x=myPolar.getSpeedData().to(speed_units),
                        y=myPolar.getSinkData().to(sink_units),
                        mode='markers',
                        name='Polar Data',)
    polarGraph.add_trace(trace1)

    # Graph the fit to the data on the same graph
    trace2 = go.Scatter(x=dfFit['Speed'].pint.to(speed_units).pint.magnitude,
                        y=dfFit['Sink'].pint.to(sink_units).pint.magnitude,
                        name=f"Fit, degree={degree}")
    polarGraph.add_trace(trace2)

    # Graph the residuals (difference between the data and the fit)
    speed_data = myPolar.getSpeedData().to(speed_units)
    sink_fit = myPolar.Sink(myPolar.getSpeedData().magnitude)
    resid = myPolar.getSinkData().to(sink_units) - (sink_fit * ureg('mps')).to(sink_units)
    trace3 = go.Scatter(x=speed_data.magnitude,
                        y=resid.magnitude,
                        name=f"Residuals")
    polarGraph.add_trace(trace3, secondary_y=True)

    # Add the weight-adjusted polar, but only if the all-up weight differs from the reference weight
    if (wFactor != 1.0):
        trace4 = go.Scatter(x=dfFit['Speed'].pint.to(speed_units).pint.magnitude * wFactor,
                            y=dfFit['Sink'].pint.to(sink_units).pint.magnitude * wFactor,
                            name=f"Fit, weight adjusted, weight {reference_weight}")
        polarGraph.add_trace(trace4)

    polarGraph.update_layout(
        xaxis_title=f"Speed ({selectedUnits['Speed'].units:~P})",
        yaxis_title=f"Sink ({selectedUnits['Sink'].units:~P})",
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
        side="right"
        )

    )

    polarGraph.update_yaxes(tickformat=".1f", secondary_y=False) 

    # Graph Speed-to-Fly vs. MC setting
    # MacCready values for table, zero to 6 knots, but must be coverted to m/s
    mcTable =  (np.arange(start=0.0, stop=6.1, step=0.02) * ureg.knots).to('mps')
    dfMc = myPolar.MacCready(mcTable)
    v =dfFit['Speed'].pint.magnitude

    goalFunctionValues = myPolar.goal_function(v, maccready.magnitude)
    traceGoal = go.Scatter(x=dfFit['Speed'].pint.to(speed_units).pint.magnitude,
                        y=goalFunctionValues,
                        name='Goal Function',)
    polarGraph.add_trace(traceGoal)

    stfGraph = make_subplots(specs=[[{"secondary_y": True}]])
    trace4 = go.Scatter(x = dfMc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = dfMc['STF'].pint.to(speed_units).pint.magnitude, 
                        name='Speed-to-Fly',
                        mode='lines')
    stfGraph.add_trace(trace4, secondary_y=False,)

    trace5 = go.Scatter(x = dfMc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = dfMc['Vavg'].pint.to(speed_units).pint.magnitude, 
                        name='Average Speed',
                        mode='lines')
    stfGraph.add_trace(trace5, secondary_y=False,)

    trace6 = go.Scatter(x = dfMc['MC'].pint.to(sink_units).pint.magnitude, 
                        y = dfMc['goalValue'],
                        name='Goal Value',
                        mode='lines')
    stfGraph.add_trace(trace6, secondary_y=True,)

    stfGraph.update_layout(
        xaxis_title=f"MacCready Setting ({selectedUnits['Sink'].units:~P})",
        yaxis_title=f"Speed ({selectedUnits['Speed'].units:~P})",
        title={
        'text': f"MacCready Speed-to-Fly",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        yaxis2=dict(
        title="Goal Value", # Title for the second (right) axis
        overlaying="y",
        side="right"
        )
    )

    if (selectedUnits['Sink'] == ureg('mps')):
        # MacCready values for table, in m/s
        mcTable =  np.arange(start=0.0, stop=3.1, step=0.5) * ureg.mps
    else:
        # MacCready values for table in knots, but must be coverted to m/s
        mcTable =  (np.arange(start=0.0, stop=6.1, step=1.0) * ureg.knots).to(ureg.mps)
    
    dfMc = myPolar.MacCready(mcTable)

    dfMc['MC'] = dfMc['MC'].pint.to(selectedUnits['Sink']).pint.magnitude
    dfMc['STF'] = dfMc['STF'].pint.to(selectedUnits['Speed']).pint.magnitude
    dfMc['Vavg'] = dfMc['Vavg'].pint.to(selectedUnits['Speed']).pint.magnitude
    
    reference_weight = current_glider['referenceWeight'].iloc[0]
    empty_weight = current_glider['emptyWeight'].iloc[0]
    pilot_weight = reference_weight - empty_weight

    print(myPolar.messages())
    print('update_graph return')
    return (glider_name, 
            myPolar.messages(), 
            reference_weight, 
            empty_weight, 
            f"{pilot_weight:,.1f}", 
            polarGraph, 
            stfGraph, 
            dfMc.to_dict('records'), 
            degree)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)