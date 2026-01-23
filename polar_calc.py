import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import fsolve
from scipy import stats

import pint
import pint_pandas
import logging

import glider

# Get access to the one-and-only UnitsRegistry instance
from units import ureg
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
logger = logging.getLogger(__name__)

class Polar:
    def __init__(self, current_glider:glider.Glider, degree, goal, v_air_horiz, v_air_vert, pilot_weight=None):
        """
        Initialize a Polar model from a Glider object and flight conditions.
        
        Parameters:
            current_glider (glider.Glider): Glider object providing reference and empty weights and polar data.
            degree (int): Polynomial degree to fit the sink-rate (polar) curve.
            goal (str): Identifier for the STF objective to use (e.g., 'Reichmann', 'Test').
            v_air_horiz (Quantity or float): Horizontal component of ambient air velocity (units expected by caller).
            v_air_vert (Quantity or float): Vertical component of ambient air velocity (units expected by caller).
            pilot_weight (optional, Quantity or float): Pilot weight to add to empty weight; if omitted, the glider's reference weight is used.
        
        Behavior:
            - Computes flight weight from glider.emptyWeight() plus pilot_weight when provided, otherwise uses glider.referenceWeight().
            - Computes a weight-scaling factor (sqrt(flight_weight / referenceWeight)); if reference weight is zero, records an error message and uses a factor of 1.0.
            - Stores configuration and computed values on the instance and initializes an internal message buffer.
            - Fits the polar polynomial by calling fit_polar(degree) which loads polar data from the glider.
        
        Side effects:
            Updates instance state (glider, goal selection, ambient air components, flight weight, weight factor, messages) and fits the sink-rate polynomial.
        """
        self.__glider = current_glider
        self.__goal_selection = goal
        self.__v_air_horiz = v_air_horiz
        self.__v_air_vert = v_air_vert
#        self.__ref_weight = current_glider['referenceWeight'].iloc[0]
#        self.__empty_weight = current_glider['emptyWeight'].iloc[0]
        self.__weight_fly = current_glider.emptyWeight() + pilot_weight*ureg('kg') if pilot_weight is not None else current_glider.referenceWeight()

        self.__messages = ""

        # Weight factor used to scale the polar sink rates
        if current_glider.referenceWeight() == 0:
            msg = "Reference weight is zero; forcing weight factor to 1.0"
            logger.error(msg)
            self.__messages += msg + "\n"
            self.__weight_factor = 1.0
        else:
            self.__weight_factor = np.sqrt(self.__weight_fly/current_glider.referenceWeight())

        self.fit_polar(degree)

    def messages(self):
        """
        Retrieve the accumulated status and error messages for this Polar instance.
        
        Returns:
            str: Concatenated message string collected during operations; may be empty.
        """
        return self.__messages
    
    # def get_reference_pilot_weight(self):
    #     return self.__ref_weight - self.__empty_weight
    
    def get_weight_fly(self):
        """
        Return the computed flight weight used for performance calculations.
        
        @returns
            flight_weight (pint.Quantity): The aircraft's flight weight (includes pilot if provided), expressed as a quantity with units.
        """
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
        w = self.__weight_factor.magnitude
        return w * self.__sink_poly(v/w) + self.__v_air_vert

    def sink_deriv(self, v):
        """
        Compute the derivative of the sink rate with respect to true airspeed, adjusted by the configured weight factor.
        
        Parameters:
            v (float): True airspeed at which to evaluate the derivative (same units as polar fit input).
        
        Returns:
            float: Value of d(sink)/d(v) at the given speed, scaled by the instance weight factor.
        """
        w = self.__weight_factor.magnitude
        return self.__sink_deriv_poly(v/w)

    # Average cross-country speed accounting for thermalling time
    # using Eq (2) from the included document "MacCready Speed to Fly Theory.pdf"
    def v_avg(self, v, mc):
        x = self.__v_air_horiz
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
        Compute the derivative (d/dV)Vavg(V) used to select the speed-to-fly that 
        maximizes average cross-country speed, accounting for thermal drift.
        
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
        ax = self.__v_air_horiz

        return (s - ((1-f)*ax + v)*s_deriv - mc) / (mc - s)**2

    # Fit the polar data to a polynomial
    # degree: the degree (order) of the polynomial to use
    def fit_polar(self, degree):
        """
        Fit the glider's sink-rate polar data to a polynomial and record fit-quality metrics.
        
        Fits speed vs. sink data from the associated glider to a polynomial of the given degree, stores the fitted polynomial and its derivative on the instance, and appends fit-quality metrics (Pearson R^2 and mean squared error) to the instance message log. For degree == 2 the fit ignores data at speeds below the minimum-sink point; for other degrees the full dataset is used.
        
        Parameters:
            degree (int): Polynomial degree to fit to the sink-rate data.
        """
        self.__degree = degree        
        speed, sink = self.__glider.polar_data_magnitude()

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
        r_value, _p_value = stats.pearsonr(sink[start_index:], sink_predicted[start_index:])

        self.__messages += f"R<sup>2</sup> = {r_value**2:.5}\n"

        # Compute mean squared error (defensively extract SSE)
        n_data_points = len(speed) - start_index
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
    