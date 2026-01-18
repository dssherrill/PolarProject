import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import fsolve
from scipy import stats

import pint
import pint_pandas
import logging

# Get access to the one-and-only UnitsRegistry instance
from units import ureg
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
logger = logging.getLogger(__name__)

class Polar:
    def __init__(self, current_glider, degree, goal, v_air_horiz, v_air_vert, pilot_weight=None):
        """
        Create a Polar model configured from a glider dataset and flight conditions.
        
        Parameters:
            current_glider (pandas.DataFrame): Glider record containing at least the keys
                'referenceWeight', 'emptyWeight', and 'polarFileName' in its first row.
            degree (int): Polynomial degree to use when fitting the polar sink-rate curve.
            goal (str): Goal selection identifier used by goal_function (e.g., 'Reichmann', 'Test').
            v_air_horiz: Horizontal component of airspeed (units/quantity expected by the caller).
            v_air_vert: Vertical component of airspeed (units/quantity expected by the caller).
            pilot_weight (optional): Pilot weight to use in flight weight calculation. If omitted,
                the reference weight from current_glider is used; if provided, flight weight is
                computed as empty weight plus pilot_weight.
        
        Side effects:
            - Stores configuration and computed values (reference/empty weight, flight weight,
              weight factor, goal selection, airspeed components) on the instance.
            - Initializes an internal messages string.
            - Loads polar data from the CSV referenced by current_glider and fits the polar
              polynomial using the specified degree.
        """
        self.__glider = current_glider
        self.__goal_selection = goal
        self.__v_air_horiz = v_air_horiz
        self.__v_air_vert = v_air_vert
        self.__ref_weight = current_glider['referenceWeight'].iloc[0]
        self.__empty_weight = current_glider['emptyWeight'].iloc[0]
        if pilot_weight is None:
            self.__weight_fly = self.__ref_weight
        else:
            self.__weight_fly = self.__empty_weight + pilot_weight

        self.__weight_factor = np.sqrt(self.__weight_fly/self.__ref_weight)

        self.__messages = ""

        self.load_CSV(current_glider['polarFileName'].iloc[0])

        self.fit_polar(degree)

    def messages(self):
        """
        Return the accumulated status and error messages recorded by this Polar instance.
        
        Returns:
            str: Concatenated message string collected during operations (may be empty).
        """
        return self.__messages
    
    def get_reference_pilot_weight(self):
        return self.__ref_weight - self.__empty_weight
    
    def get_weight_fly(self):
        return self.__weight_fly
    
    def get_weight_factor(self):
        return self.__weight_factor
    
    def goal_function(self, v, mc):
        """
        Dispatches to the selected goal-specific objective function.
        
        Parameters:
            v (float): Airspeed (m/s) at which to evaluate the objective.
            mc (float): MacCready value (m/s) used by the objective.
        
        Returns:
            float: The objective function residual computed for the chosen goal.
        
        Raises:
            ValueError: If the configured goal selection is not "Reichmann" or "Test".
        """
        g = self.__goal_selection
        if g == 'Reichmann':
            return self.goal_function_1(v, mc)
        elif g == 'Test':
            return self.goal_function_2(v, mc)
        else:
            raise ValueError(f'Goal selection is {g} but must be "Reichmann" or "Test".')
            
    def sink(self, v):
        """
        Compute the glider's sink rate at a given true airspeed, scaled for the configured weight and adjusted by the ambient vertical airspeed.
        
        Parameters:
            v (float): True airspeed at which to evaluate the polar, in meters per second.
        
        Returns:
            float: Sink rate in meters per second (negative values indicate climb, positive values indicate descent).
        """
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
        return s - v * s_deriv - mc

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
        """
        Compute the normalized objective residual used to select the speed-to-fly that maximizes average cross-country speed, accounting for thermal drift.
        
        Parameters:
            v (float): True airspeed at which to evaluate the objective (m/s).
            mc (float): MacCready setting (m/s).
        
        Returns:
            float: Objective residual value; a root (zero) indicates an airspeed v that satisfies the maximum-average-speed condition.
        """
        f = 1.0  

        # self.sink(v) includes Wm = self.__v_air_vert
        s = self.sink(v)
        s_deriv = self.sink_deriv(v)
        ax = self.__v_air_horiz.magnitude

        return (s - ((1-f)*ax + v)*s_deriv - mc) / (mc - s)**2

    # Fit the polar data to a polynomial
    # degree: the degree (order) of the polynomial to use
    def fit_polar(self, degree):
        """
        Fit the polar sink-rate data to a polynomial and record fit-quality metrics.
        
        Fits the instance's loaded speed and sink data to a polynomial of the given degree. For degree == 2 the fit starts at the speed corresponding to the minimum sink; otherwise the full dataset is used. The fitted polynomial and its derivative are stored on the instance, and the method computes predicted sink values to derive and append R^2 and mean squared error (MSE) information to the instance message log.
        
        Parameters:
            degree (int): Polynomial degree to fit to the sink-rate data.
        """
        self.__degree = degree        
        speed = self.__speed_data.magnitude
        sink = self.__sink_data.magnitude

        # Low-order fits should ignore polar data at speeds below minimum sink
        # because the model cannot follow the curvature near stall speed
        min_sink_index = np.argmax(sink)
        if degree == 2:
            start_index = min_sink_index
        else:
            start_index = 0

        self.__sink_poly, (SSE, _rank, _sv, _rcond) = Poly.fit(speed[start_index:], sink[start_index:], degree, full=True)
        self.__sink_deriv_poly = self.__sink_poly.deriv()

        # Generate predicted y-values
        sink_predicted = self.__sink_poly(speed)

        # Calculate the R-value (Pearson correlation coefficient)
        r_value, _p_value = stats.pearsonr(speed, sink_predicted)

        self.__messages += f"R<sup>2</sup> = {r_value**2:.3}\n"

        # Compute mean squared error (defensively extract SSE)
        n_data_points = len(speed)
        if isinstance(SSE, (list, tuple, np.ndarray)) and len(SSE) > 0:
            SSE_val = SSE[0]
        else:
            try:
                SSE_val = float(SSE)
            except Exception:
                SSE_val = 0.0

        MSE = SSE_val / n_data_points
        self.__messages += f"MSE = {MSE:.3}"

    def Sink(self, speed):
        return self.__sink_poly(speed)

    # mcTable has MacCready values in units of m/s
    def MacCready(self, mcTable):
        """
        Compute MacCready performance table for a sequence of MacCready values.
        
        Parameters:
            mcTable (array-like of pint.Quantity): Sequence of MacCready settings (values in meters per second).
        
        Returns:
            pandas.DataFrame: DataFrame with columns
                - MC: MacCready values as quantities in m/s
                - STF: Optimal speed-to-fly at each MC as quantities in m/s
                - Vavg: Net cross-country average speed (accounts for thermalling) as quantities in m/s
                - L/D: Lift-to-drag ratio at the STF (dimensionless)
                - solverResult: Solver residual or goal-function value for the found STF
        """
        Vstf = np.zeros(len(mcTable))   # optimum speed-to-fly
        Vavg = np.zeros(len(mcTable))   # net cross-country speed, taking thermalling time into account
        LD = np.zeros(len(mcTable))     # L/D ratio at Vstf
        solver_result = np.zeros(len(mcTable))

        # Guess 50 knots, but must express as m/s
        initial_guess = ureg('50.0 knots').to(ureg.mps).magnitude

        # For each MC value, find the speed at which "goal_function" is equal to zero
        for i in range(len(mcTable)):
            mc = mcTable[i]
            [solution, _, err, msg] = fsolve(self.goal_function, initial_guess, (mc.magnitude), full_output=True, xtol=1e-5)
            if err == 1:
                v = solution[0]
                Vstf[i] = v

                # Reichmann: Vcruise = V * Cl / (Cl - Si); Cl = climb rate (positive); Si = sink rate (negative)
                if len(solution) > 1:
                    self.__messages += f'{i=}, {v=}, {len(solution)=}\n'

                sink = self.sink(v)
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

    # Reads a polar from a CSV file created by WebPlotDigitizer at https://automeris.io/
    # x = first column  = speed in km per hour
    # y = second column = sink in m per second (all values should be negative)
    def load_CSV(self, polar_file_name):
        """
        Load polar speed and sink data from a CSV file in ./datafiles and store them as pint-quantified NumPy arrays.
        
        Parameters:
            polar_file_name (str): CSV file name located in ./datafiles/; expected format: first column = speed in km/h, second column = sink in m/s.
        
        Description:
            - On success populates self.__speed_data with speeds converted to meters per second and self.__sink_data with sink rates as meters per second.
            - On FileNotFoundError sets both __speed_data and __sink_data to None and appends an error message to self.__messages.
            - On any other exception sets both __speed_data and __sink_data to None and appends the exception message to self.__messages.
            - Raises ValueError if the loaded speed or sink arrays are missing or empty after reading the CSV.
        """
        logger.debug(f'polarFileName is "{polar_file_name}"')
        file_path = f"./datafiles/{polar_file_name}"
        logger.debug(f'file_path is "{file_path}"')

        try:
            df_polar = pd.read_csv(file_path)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            # Explicitly mark data as not loaded so callers can detect failure
            self.__speed_data = None
            self.__sink_data = None
            return None
        except Exception as e:
            # Catch any exceptions, record message and return early to avoid using undefined df_polar
            self.__messages += f"An unexpected error occurred: {e}\n"
            self.__speed_data = None
            self.__sink_data = None
            return None

        # Convert speed from km/hr to m/s
        self.__speed_data = (df_polar.iloc[:,0].to_numpy() * ureg.kph).to('mps')

        # Sink is already in m/s
        self.__sink_data = df_polar.iloc[:,1].to_numpy() * ureg.mps

        # Ensure CSV data loaded successfully before fitting
        if self.__speed_data is None or len(self.__speed_data) == 0:
            raise ValueError('Polar speed data not loaded or empty; cannot fit polar.')
        if self.__sink_data is None or len(self.__sink_data) == 0:
            raise ValueError('Polar sink data not loaded or empty; cannot fit polar.')


    def get_polar(self):
        """
        Retrieve the loaded polar speed and sink datasets.
        
        Returns:
            tuple: (speed_data, sink_data)
                speed_data: Sequence of speeds as pint `Quantity` in meters per second.
                sink_data: Corresponding sink rates as pint `Quantity` in meters per second.
        """
        return self.__speed_data, self.__sink_data
    
    def getSpeedData(self):
        return self.__speed_data
    
    def getSinkData(self):
        return self.__sink_data
    

    