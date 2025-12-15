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
    def __init__(self, currentGlider, degree, emptyWeight, pilotWeight=None):
        self.__glider = currentGlider,
        self.__wRef = currentGlider['referenceWeight'].iloc[0]
        self.__wEmpty = currentGlider['emptyWeight'].iloc[0]
        if (pilotWeight == None):
            pilotWeight = self.__wRef - self.__wEmpty

        self.__wFly = self.__wEmpty + pilotWeight
        self.__wFactor = np.sqrt(self.__wFly/self.__wRef)

        self.__messages = ""

        print(f'ref {self.__wRef}')
        print(f'fly {self.__wFly}')

#        self.loadJSON(currentGlider['polarFileName'].iloc[0])
        self.loadCSV(currentGlider['polarFileName'].iloc[0])
        self.fitPolar(degree)

    def message(self):
        return self.__messages
    
    def get_wFactor(self):
        return self.__wFactor

    def fitPolar(self, degree):
        self.__degree = degree
        
        speed = self.__speedData.magnitude
        sink = self.__sinkData.magnitude

        self.__sinkPoly, (residuals, rank, sv, rcond) = Poly.fit(speed, sink, degree, full=True)

        # Generate predicted y-values
        sink_predicted = self.__sinkPoly(speed)

        # Calculate the R-value (Pearson correlation coefficient)
        r_value, p_value = stats.pearsonr(speed, sink_predicted)

        self.__messages = f"R = {r_value:.3}\n"

        # The 'residuals' is the sum of squared errors
        chi_squared_value = residuals[0]
        self.__messages += f"Chi-squared = {chi_squared_value:.3}\n"

        # To get the reduced chi-squared (chi-squared per degree of freedom)
        n_data_points = len(speed)
        n_parameters = degree + 1 # A degree N polynomial has N+1 coefficients (parameters)
        degrees_of_freedom = n_data_points - n_parameters

        if degrees_of_freedom > 0:
            reduced_chi_squared = chi_squared_value / degrees_of_freedom
            self.__messages += f"Reduced Chi-squared = {reduced_chi_squared:.3}\n"
        else:
            self.__messages += f"Reduced chi-squared is not defined when degrees of freedom is less than 1;  DOF = {degrees_of_freedom}.\n"
        
    def Sink(self, speed):
        return self.__sinkPoly(speed)

    # mcTable has MacCready values in units of m/s
    def MacCready(self, mcTable):

        # Reichmann STF equation (Equation II) after moving left-hand-side to right-hand-side
        #  0 = Ws + Wm - Cl - (dWs/dV)V 
        #  V  = glider's speed
        #  Ws = sink rate from the glider's polar (negative)
        #  Wm = air mass movement = 0 for now
        #  Cl = climb rate (positive) = MacCready value

        # v = speedPoly = speed in m/s
        # mc = MacCready setting in m/s
        # w = adjustment factor for actual takeoff weight

        # compute goalFunction = sink - speed * (derivative of sink)
        sinkDeriv = self.__sinkPoly.deriv()
        self.__goalFunction = lambda v, mc, w:  w*self.__sinkPoly(v/w) - (v/w) * w*sinkDeriv(v/w) - mc 

        root = np.zeros(len(mcTable))
        Vavg = np.zeros(len(mcTable))

        # Guess 80 knots, but must express as m/s
        initial_guess = ureg('80.0 knots').to(ureg.mps).magnitude

        # For each MC value, find the speed at which "goalFunction" is equal to zero
        for i in range(len(mcTable)):
            mc = mcTable[i]
            [solution, d, err, msg] = fsolve(self.__goalFunction, initial_guess, (mc.magnitude, self.__wFactor), full_output=True)
            if err == 1:
                root[i] = solution[0]
                # Reichmann: Vcruise = V * Cl / (Cl - Si); Cl = climb rate (positive); Si = sink rate (negative)
                w = self.__wFactor
                v = root[i]  #  v = STF / weight adjustment factor
                if len(solution) > 1:
                    self.__messages += f'{i=}, {v=}, {len(solution)=}\n'

                sink = w * self.__sinkPoly(v/w)
                Vavg[i] = v * mc.magnitude / (mc.magnitude - sink)  

                # Use this solution as the initial guess for the next v value 
                initial_guess = v
            else:
                self.__messages += f"Solution not found for index {i}, MC = {mc:0.3f} m/s\n"
                self.__messages += f"Reason: {msg}\n"
                
        dfMc = pd.DataFrame({'MC':  PA_(mcTable, ureg.mps),
                            'STF':  PA_(root, ureg.mps),
                            'Vavg': PA_(Vavg, ureg.mps)
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
    def loadCSV(self, polarFileName):
        print(f'polarFileName is "{polarFileName}"')
        file_path = f"./datafiles/{polarFileName}"
        print(f'file_path {file_path}')

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            return None
        except Exception as e:
            # Catch any exceptions
            self.__messages += f"An unexpected error occurred: {e}\n"

        # Convert speed from km/hr to m/s
        self.__speedData = (df.iloc[:,0].to_numpy() * ureg.kph).to('mps')

        # Sink is already in m/s
        self.__sinkData = df.iloc[:,1].to_numpy() * ureg.mps

    def get_polar(self):
        return self.__speedData, self.__sinkData
    
    def getSpeedData(self):
        return self.__speedData
    
    def getSinkData(self):
        return self.__sinkData
    

    
# glider="Duo Discus T"
# glider="ASW 28"
# df = loadPolar(glider)
# plt.scatter(x=df['Speed'], y=df['Sink'])
# plt.show()