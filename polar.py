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
    def __init__(self, current_glider, degree, goal, v_air_horiz, v_air_vert, pilot_weight=None):
        self.__glider = current_glider
        self.__goal_selection = goal
        self.__v_air_horiz = v_air_horiz
        self.__v_air_vert = v_air_vert
        self.__ref_weight = current_glider['referenceWeight'].iloc[0]
        self.__empty_weight = current_glider['emptyWeight'].iloc[0]
        if (pilot_weight == None):
            self.__weight_fly = self.__ref_weight
        else:
            self.__weight_fly = self.__empty_weight + pilot_weight

        self.__weight_factor = np.sqrt(self.__weight_fly/self.__ref_weight)

        self.__messages = ""

#       self.load_JSON(currentGlider['polarFileName'].iloc[0])
        self.load_CSV(current_glider['polarFileName'].iloc[0])
        self.fit_polar(degree)

    def messages(self):
        return self.__messages
    
    def get_reference_pilot_weight(self):
        return self.__ref_weight - self.__empty_weight
    
    def get_weight_fly(self):
        return self.__weight_fly
    
    def get_weight_factor(self):
        return self.__weight_factor
    
    def goal_function(self, v, mc):
        g = self.__goal_selection
        if g == 'Reichmann':
            return self.goal_function_1(v, mc)
        else:
            if g == 'Mine':
                return self.goal_function_2(v, mc)
            else:
                raise ValueError(f'Goal selection is {g} but must be "Reichmann" or "Mine".')
            
    def sink(self, v):
        w = self.__weight_factor
        return w * self.__sink_poly(v/w) + self.__v_air_vert.magnitude

    def sink_deriv(self, v):
        w = self.__weight_factor
        return self.__sink_deriv_poly(v/w)

    def v_avg(self, v, mc):
        x = self.__v_air_horiz.magnitude
        f = 1.0
        s = self.sink(v)
        return (mc*(v+x) - f*x*s)/(mc - s)

    def goal_function_1(self, v, mc):
        # Reichmann STF equation (Equation II) after moving left-hand-side to right-hand-side
        #  0 = Ws + Wm - Cl - (dWs/dV)V 
        #  V  = glider's airspeed speed
        #  Ws = sink rate from the glider's polar (negative)
        #  Wm = vertical air mass movement when cruising
        #  Cl = climb rate when thermalling (positive) = MacCready value

        # v = speed in m/s
        # mc = MacCready setting in m/s
        # wf = adjustment factor for actual takeoff weight
        # self.sink(v) includes Wm = self.__v_air_vert
        s = self.sink(v)
        s_deriv = self.sink_deriv(v)
        return s + - v * s_deriv - mc

    # goal_function_1 uses Reichmann's equation for minimum time
    # goal_function_2 solves for maximum average speed
    # This model seems to give different results when the airmass has vertical motion (during cruise) 
    def goal_function_2(self, v, mc):
        # f is a factor that defines the relative horizontal speed of the thermal
        # 1 means the thermal drifts at the same speed as the horizontal wind (in cruise)
        # 0 means the thermal rise vertically, with no horizontal motion
        # G Dale says that the right answer is often somewhere in between, because
        # the rising thermal has significant mass and does not immediately accelerate
        # to the speed of the wind around it
        f = 1.0  

        # self.sink(v) includes Wm = self.__v_air_vert
        s = self.sink(v)
        s_deriv = self.sink_deriv(v)
        ax = self.__v_air_horiz.magnitude

        return (mc + ((1-f)*ax + v)*s_deriv - s) # /(mc - s)**2

    # Fit the polar data to a polynomial
    # degree: the degree (order) of the polynomial to use
    def fit_polar(self, degree):
        self.__degree = degree
        
        speed = self.__speed_data.magnitude
        sink = self.__sink_data.magnitude

        self.__sink_poly, (SSE, rank, sv, rcond) = Poly.fit(speed, sink, degree, full=True)
        self.__sink_deriv_poly = self.__sink_poly.deriv()

        # Generate predicted y-values
        sink_predicted = self.__sink_poly(speed)

        # Calculate the R-value (Pearson correlation coefficient)
        r_value, p_value = stats.pearsonr(speed, sink_predicted)

        self.__messages += f"R<sup>2</sup> = {r_value**2:.3}\n"

        # Compute mean squared error
        n_data_points = len(speed)
        MSE = SSE[0] / n_data_points
        self.__messages += f"MSE = {MSE:.3}"

    def Sink(self, speed):
        return self.__sink_poly(speed)

    # mcTable has MacCready values in units of m/s
    def MacCready(self, mcTable):
        Vstf = np.zeros(len(mcTable))   # optimum speed-to-fly
        Vavg = np.zeros(len(mcTable))   # net cross-country speed, taking thermalling time into account
        LD = np.zeros(len(mcTable))     # L/D ratio at Vstf
        solver_result = np.zeros(len(mcTable))

        # Guess 80 knots, but must express as m/s
        initial_guess = ureg('50.0 knots').to(ureg.mps).magnitude

        # For each MC value, find the speed at which "goal_function" is equal to zero
        wf = self.__weight_factor
        for i in range(len(mcTable)):
            mc = mcTable[i]
            [solution, d, err, msg] = fsolve(self.goal_function, initial_guess, (mc.magnitude), full_output=True, xtol=1e-5)
            if err == 1:
                v = solution[0]
                Vstf[i] = v

                # Reichmann: Vcruise = V * Cl / (Cl - Si); Cl = climb rate (positive); Si = sink rate (negative)
                if len(solution) > 1:
                    self.__messages += f'{i=}, {v=}, {len(solution)=}\n'

                sink = wf * self.__sink_poly(v/wf) + self.__v_air_vert.magnitude
                LD[i] = -v / sink # negative sign because sink in negative by L/D is always express as positive
#                Vavg[i] = (v * mc.magnitude / (mc.magnitude - sink))
                Vavg[i] = self.v_avg(v, mc.magnitude)
                solver_result[i] = self.goal_function(v, mc.magnitude,)

                # Use this solution as the initial guess for the next v value 
                initial_guess = v
            else:
                self.__messages += f"\nSolution not found for index {i}, MC = {mc:0.3f} m/s\n"
                self.__messages += f"Reason: {msg}\n"
                
        df_mc = pd.DataFrame({'MC':  PA_(mcTable, ureg.mps),
                            'STF':  PA_(Vstf, ureg.mps),
                            'Vavg': PA_(Vavg, ureg.mps),
                            'L/D': LD,
                            'solverResult': solver_result
                            })
        return df_mc
    
    # Throughout this code, the conventional units are
    #  m/s for speed and 
    #  kg for weight

    # Reads a polar from a JSON file created by WebPlotDigitizer at https://automeris.io/
    # x = speed in km per hour
    # y = sink in m per second (all values should be negative)
    def load_JSON(self, polarFileName):
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

        json_norm = pd.json_normalize(jsonData['datasetColl'][0]['data'])
        json_polar_data = json_norm['value']
        speed, sink = map(list, zip(*json_polar_data))

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
            df_polar = pd.read_csv(file_path)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            return None
        except Exception as e:
            # Catch any exceptions
            self.__messages += f"An unexpected error occurred: {e}\n"

        # Convert speed from km/hr to m/s
        self.__speed_data = (df_polar.iloc[:,0].to_numpy() * ureg.kph).to('mps')

        # Sink is already in m/s
        self.__sink_data = df_polar.iloc[:,1].to_numpy() * ureg.mps

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