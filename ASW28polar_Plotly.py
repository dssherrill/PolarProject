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
defaultPolynomialDegree = 5
wFactor = 1.0

# display options
metricUnits = {'Speed': 'kph', 'Sink': 'm/s'}
UsUnits = {'Speed': 'knots', 'Sink': 'knots'}
unitLabels = {'Metric' : metricUnits, 'knots' : UsUnits}

def polarCalc(currentGlider, degree, emptyWeight, pilotWeight):
    gliderName = currentGlider['name'].iloc[0]
    myPolar = polar.Polar(currentGlider, degree, emptyWeight, pilotWeight)
    speed, sink = myPolar.get_polar()

    # Evaluate the polynomial for new points
    speed = np.linspace(min(speed).magnitude, max(speed).magnitude, 100)
    dfFit = pd.DataFrame({'Speed': PA_(speed, ureg.mps), 'Sink': PA_(myPolar.Sink(speed), ureg.mps)})
    
    return dfFit, myPolar

# dummy data to setup the AG Grid as a MacCready table
initial_data = pd.DataFrame({'MC': [0], 'STF': [0], 'Vavg': [0]})

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
                    value=dropdown_options[0]['value'] if dropdown_options else None, # Set a default value if options exist
                    placeholder="Select a glider...",
                ),
                html.Div(id="output-container"),
                            
            ], className="mb-3"),

            

            dbc.Row([
                dbc.Row([
                #--- reference weight
                    dbc.Label("Reference weight (kg):", html_for="ref-weight-input"),
                    dcc.Input(id="ref-weight-input", type="number", placeholder="400"),

                #--- empty weight
                    dbc.Label("Empty weight (kg):", html_for="empty-weight-input"),
                    dcc.Input(id="empty-weight-input", type="number", placeholder="300"),

                #--- reference weight
                    dbc.Label("Pilot + Ballast weight (kg):", html_for="pilot-weight-input"),
                    dcc.Input(id="pilot-weight-input", type="number", placeholder="100", debounce=True),
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
            dbc.Row([
                dbc.Col([
                #--- Polynomial degree
                    dbc.Label("Polynomial degree:", html_for="poly-degree"),
                    dcc.Input(id="poly-degree", type="number", placeholder=defaultPolynomialDegree, style={"width": "2"}),
                    ], width=2),
            ]),

            # Statistics output group
            dbc.Row([
                html.Div("Statistics", style={"whiteSpace": "pre-line", "fontFamily": "Courier, serif"}, id="statistics"),
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
    State(component_id='ref-weight-input', component_property='value'),
    State(component_id='empty-weight-input', component_property='value'),
    State(component_id='pilot-weight-input', component_property='value'),
#    prevent_initial_call=True
)
def update_graph(n_clicks, degree, col_chosen, units, referenceWeight, emptyWeight, pilotWeight):
    currentGlider = dfGliderInfo[dfGliderInfo['name'] == col_chosen]

    if degree is None:
        degree = defaultPolynomialDegree

    # The model must be a quadratic or higher order polynomial
    degree = max(degree, 2)

    dfFit, myPolar = polarCalc(currentGlider, degree, emptyWeight, pilotWeight)
    wFactor = myPolar.get_wFactor()

    # handle display units option
    selectedUnits = unitLabels[units]
    sink_units = ureg(selectedUnits['Sink'])
    speed_units = ureg(selectedUnits['Speed'])

    # Graph the polar data
    polarGraph = go.Figure()
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

    # Add the weight-adjusted polar, but only if a weight was specified
    if (wFactor != 1.0):
        trace3 = go.Scatter(x=dfFit['Speed'].pint.to(speed_units).pint.magnitude * wFactor,
                            y=dfFit['Sink'].pint.to(sink_units).pint.magnitude * wFactor,
                            name=f"Fit, weight adjusted, weight {referenceWeight}")
        polarGraph.add_trace(trace3)

    polarGraph.update_layout(
        xaxis_title=f"Speed ({selectedUnits['Speed']})",
        yaxis_title=f"Sink ({selectedUnits['Sink']})"
    )

    polarGraph.update_yaxes(tickformat=".1f")

    # Graph Speed-to-Fly vs. MC setting
    # MacCready values for table, zero to 6 knots, but must be coverted to m/s
    mcTable =  (np.arange(start=0.0, stop=6.1, step=0.02) * ureg.knots).to('mps')
    dfMc = myPolar.MacCready(mcTable)

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

    stfGraph.update_layout(
        xaxis_title=f"MacCready ({selectedUnits['Sink']})",
        yaxis_title=f"Speed ({selectedUnits['Speed']})",
    )

    if (selectedUnits['Sink'] == 'm/s'):
        # MacCready values for table, in m/s
        mcTable =  np.arange(start=0.0, stop=3.1, step=0.5) * ureg.mps
    else:
        # MacCready values for table in knots, but must be coverted to m/s
        mcTable =  (np.arange(start=0.0, stop=6.1, step=1.0) * ureg.knots).to(ureg.mps)
    
    dfMc = myPolar.MacCready(mcTable)

    dfMc['MC'] = dfMc['MC'].pint.to(selectedUnits['Sink']).pint.magnitude
    dfMc['STF'] = dfMc['STF'].pint.to(selectedUnits['Speed']).pint.magnitude
    dfMc['Vavg'] = dfMc['Vavg'].pint.to(selectedUnits['Speed']).pint.magnitude
    
    reference_weight = currentGlider['referenceWeight'].iloc[0]
    empty_weight = currentGlider['emptyWeight'].iloc[0]
    pilot_weight = reference_weight - empty_weight

    """ 
    # Check if the input value is None or empty before proceeding
    if pilotWeight is None or pilotWeight == "":
        raise PreventUpdate
    """    

    print(myPolar.message())
    print('update_graph return')
    return (col_chosen, 
            myPolar.message(), 
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