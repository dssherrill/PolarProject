import numpy as np
import pandas as pd
import json

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import fsolve
from scipy import stats

import pint
import pint_pandas

# Get access to the one-and-only UnitsRegistry instance
from units import ureg
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity

class Polar:
    def __init__(self, current_glider, degree, v_air_horiz, v_air_vert, pilot_weight=None):
        self.__glider = current_glider
        self.__v_air_horiz = v_air_horiz
        self.__v_air_vert = v_air_vert
        self.__ref_weight = current_glider['referenceWeight'].iloc[0]
        self.__empty_weight = current_glider['emptyWeight'].iloc[0]
        if (pilot_weight == None):
            pilot_weight = self.__ref_weight - self.__empty_weight

        self.__wFly = self.__empty_weight + pilot_weight
        self.__weight_factor = np.sqrt(self.__wFly/self.__ref_weight)

        self.__messages = ""

#       self.loadJSON(currentGlider['polarFileName'].iloc[0])
        self.load_CSV(current_glider['polarFileName'].iloc[0])
        self.fit_polar(degree)

    def messages(self):
        return self.__messages
    
    def get_weight_factor(self):
        return self.__weight_factor
    
    def goal_function(self, v, mc):
        return self.__goal_function(v, mc)

    def fit_polar(self, degree):
        self.__degree = degree
        
        speed = self.__speed_data.magnitude
        sink = self.__sink_data.magnitude

        self.__sink_poly, (residuals, rank, sv, rcond) = Poly.fit(speed, sink, degree, full=True)
        self.__sinkDeriv = self.__sink_poly.deriv()

        # Generate predicted y-values
        sink_predicted = self.__sink_poly(speed)

        # Calculate the R-value (Pearson correlation coefficient)
        r_value, p_value = stats.pearsonr(speed, sink_predicted)

        self.__messages = f"R = {r_value:.3}\n"

        # The 'residuals' is the sum of squared errors
        chi_squared_value = residuals[0]
        self.__messages += f"&chi;<sup>2</sup> = {chi_squared_value:.3}\n"

        # To get the reduced chi-squared (chi-squared per degree of freedom)
        n_data_points = len(speed)
        n_parameters = degree + 1 # A degree N polynomial has N+1 coefficients (parameters)
        degrees_of_freedom = n_data_points - n_parameters

        if degrees_of_freedom > 0:
            reduced_chi_squared = chi_squared_value / degrees_of_freedom
            self.__messages += f"Reduced &chi;<sup>2</sup> = {reduced_chi_squared:.3}\n"
        else:
            self.__messages += f"Reduced &chi;<sup>2</sup> is not defined when degrees of freedom is less than 1;  DOF = {degrees_of_freedom}.\n"
        
    def Sink(self, speed):
        return self.__sink_poly(speed)

    # mcTable has MacCready values in units of m/s
    def MacCready(self, mcTable):
        # Reichmann STF equation (Equation II) after moving left-hand-side to right-hand-side
        #  0 = Ws + Wm - Cl - (dWs/dV)V 
        #  V  = glider's airspeed speed
        #  Ws = sink rate from the glider's polar (negative)
        #  Wm = air mass movement = 0 for now
        #  Cl = climb rate (positive) = MacCready value

        # v = speedPoly = speed in m/s
        # mc = MacCready setting in m/s
        # wf = adjustment factor for actual takeoff weight

        # compute goalFunction = sink - speed * (derivative of sink)
        self.__sinkDeriv = self.__sink_poly.deriv()
        self.__goal_function = lambda v, mc:  self.__weight_factor*self.__sink_poly(v/self.__weight_factor) - (v/self.__weight_factor) * self.__weight_factor*self.__sinkDeriv(v/self.__weight_factor) - mc -self.__v_air_vert.magnitude

        Vstf = np.zeros(len(mcTable))   # optimum speed-to-fly
        Vavg = np.zeros(len(mcTable))   # net cross-country speed, taking thermalling time into account
        LD = np.zeros(len(mcTable))     # L/D ratio at Vstf
        goalValue = np.zeros(len(mcTable))

        # Guess 80 knots, but must express as m/s
        initial_guess = ureg('80.0 knots').to(ureg.mps).magnitude

        # For each MC value, find the speed at which "goalFunction" is equal to zero
        wf = self.__weight_factor
        for i in range(len(mcTable)):
            mc = mcTable[i]
            [solution, d, err, msg] = fsolve(self.__goal_function, initial_guess, (mc.magnitude), full_output=True, xtol=1e-5)
            if err == 1:
                v = solution[0]
                Vstf[i] = v

                # Reichmann: Vcruise = V * Cl / (Cl - Si); Cl = climb rate (positive); Si = sink rate (negative)
                if len(solution) > 1:
                    self.__messages += f'{i=}, {v=}, {len(solution)=}\n'

                sink = wf * self.__sink_poly(v/wf)
                LD[i] = -v / sink # negative sign because sink in negative by L/D is always express as positive
                Vavg[i] = (v * mc.magnitude / (mc.magnitude - sink))
                goalValue[i] = self.__goal_function(v, mc.magnitude,)

                # Use this solution as the initial guess for the next v value 
                initial_guess = v
            else:
                self.__messages += f"Solution not found for index {i}, MC = {mc:0.3f} m/s\n"
                self.__messages += f"Reason: {msg}\n"
                
        dfMc = pd.DataFrame({'MC':  PA_(mcTable, ureg.mps),
                            'STF':  PA_(Vstf, ureg.mps),
                            'Vavg': PA_(Vavg, ureg.mps),
                            'L/D': LD,
                            'goalValue': goalValue
                            })
        return dfMc
    
    # Throughout this code, the conventional units are
    #  m/s for speed and 
    #  kg for weight

    # Reads a polar from a JSON file created by WebPlotDigitizer at https://automeris.io/
    # x = speed in km per hour
    # y = sink in m per second (all values should be negative)
    def loadJSON(self, polarFileName):
        print(f'polarFileName is "{polarFileName}"')
        file_path = f"./datafiles/{polarFileName}"
        print(f'file_path {file_path}')

        try:
            with open(file_path, 'r') as f:
                jsonData = json.load(f)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            return None
        except json.JSONDecodeError as e:
            self.__messages += f"Error decoding JSON: {e}\n"
            return None

        jsonNorm = pd.json_normalize(jsonData['datasetColl'][0]['data'])
        jsonPolarData = jsonNorm['value']
        speed, sink = map(list, zip(*jsonPolarData))

        # Convert speed from km/hr to m/s
        speed = speed * ureg.kph
        speed.ito(ureg.mps)

        # Sink is already in m/s
        sink = sink * ureg.mps

        self.__polar = pd.DataFrame({
            'Speed':pd.Series(speed, dtype = 'pint[mps]'),
            'Sink': pd.Series(sink, dtype = 'pint[mps]'),
        })


    # Reads a polar from a CSV file created by WebPlotDigitizer at https://automeris.io/
    # x = first column  = speed in km per hour
    # y = second column = sink in m per second (all values should be negative)
    def load_CSV(self, polar_file_name):
        print(f'polarFileName is "{polar_file_name}"')
        file_path = f"./datafiles/{polar_file_name}"
        print(f'file_path is "{file_path}"')

        try:
            dfPolar = pd.read_csv(file_path)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            return None
        except Exception as e:
            # Catch any exceptions
            self.__messages += f"An unexpected error occurred: {e}\n"

        # Convert speed from km/hr to m/s
        self.__speed_data = (dfPolar.iloc[:,0].to_numpy() * ureg.kph).to('mps')

        # Sink is already in m/s
        self.__sink_data = dfPolar.iloc[:,1].to_numpy() * ureg.mps

    def get_polar(self):
        return self.__speed_data, self.__sink_data
    
    def getSpeedData(self):
        return self.__speed_data
    
    def getSinkData(self):
        return self.__sink_data
    

    
# glider="Duo Discus T"
# glider="ASW 28"
# df = loadPolar(glider)
# plt.scatter(x=df['Speed'], y=df['Sink'])
# plt.show()