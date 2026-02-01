import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy import stats

import pint
import pint_pandas
import logging

import glider

# Get access to the one-and-only UnitsRegistry instance
from units import ureg

PA_ = pint_pandas.PintArray
logger = logging.getLogger(__name__)


class Polar:
    def __init__(
        self,
        current_glider: glider.Glider,
        degree,
        goal,
        v_air_horiz,
        v_air_vert,
        pilot_weight=None,
    ):
        """
        Initialize a Polar model from a glider dataset and flight conditions.

        Configures the instance with the provided glider record, goal selection, and airspeed
        components; computes the flight weight and a weight-scaling factor, records status
        messages, and fits the polar sink-rate polynomial.

        Parameters:
            current_glider (glider.Glider): Glider record providing reference and empty weights
                and polar data accessors.
            degree (int): Polynomial degree to use when fitting the sink-rate (polar) curve.
            goal (str): Goal selector used by the instance's goal_function (e.g., "Reichmann", "Test").
            v_air_horiz: Horizontal component of ambient airspeed (quantity expected by caller).
            v_air_vert: Vertical component of ambient airspeed (quantity expected by caller).
            pilot_weight (optional): Pilot weight to add to empty weight (quantity with units,
                if omitted the glider's reference weight is used).

        Side effects:
            - Stores configuration and computed weight/weight-factor on the instance.
            - Appends status messages to the instance message log.
            - Calls fit_polar(degree) to load and fit the polar data.
        """
        self.__glider = current_glider
        self.__goal_selection = goal
        self.__v_air_horiz = v_air_horiz
        self.__v_air_vert = v_air_vert
        #        self.__ref_weight = current_glider['referenceWeight'].iloc[0]
        #        self.__empty_weight = current_glider['emptyWeight'].iloc[0]
        self.__weight_fly = (
            current_glider.emptyWeight() + pilot_weight * ureg("kg")
            if pilot_weight is not None
            else current_glider.referenceWeight()
        )

        self.__messages = ""

        # Weight factor used to scale the polar sink rates
        if current_glider.referenceWeight() == 0:
            msg = "Reference weight is zero; forcing weight factor to 1.0"
            logger.error(msg)
            self.__messages += msg + "\n"
            self.__weight_factor = 1.0
        else:
            self.__weight_factor = np.sqrt(
                self.__weight_fly.magnitude / current_glider.referenceWeight().magnitude
            )

        self.fit_polar(degree)

    def messages(self):
        """
        Get accumulated status and error messages for this Polar instance.

        Returns:
            str: Concatenated message string collected during operations; empty string if no messages.
        """
        return self.__messages

    # def get_reference_pilot_weight(self):
    #     return self.__ref_weight - self.__empty_weight

    def get_weight_fly(self):
        """
        Return the computed flight weight used for performance calculations.

        Returns:
            The flight weight (includes glider empty weight plus pilot weight when provided) as a pint quantity with units of kilograms.
        """
        return self.__weight_fly

    def get_weight_factor(self):
        """
        Provide the dimensionless weight-scaling factor applied to the polar.

        Returns:
            float: Weight factor (greater than 0) used to scale speeds and sink rates, computed as sqrt(flight weight / reference weight).
        """
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
        if g == "Reichmann":
            return self.goal_function_1(v, mc)
        elif g == "Test":
            return self.goal_function_2(v, mc)
        else:
            raise ValueError(
                f'Goal selection is {g} but must be "Reichmann" or "Test".'
            )

    def sink(self, v, weight_correction=True, include_airmass=True):
        """
        Evaluate the glider's sink rate at a given true airspeed, optionally applying the configured weight scaling and ambient vertical-air adjustment.

        Parameters:
            v (float): True airspeed in meters per second.
            weight_correction (bool): If True, scale the sink rate by the configured weight factor; if False, use the unscaled polar.
            include_airmass (bool): If True, add the configured ambient vertical airspeed (positive = descent) to the sink rate; if False, omit ambient vertical motion.

        Returns:
            float: Sink rate in meters per second; negative values indicate climb, positive values indicate descent.
        """
        w = self.__weight_factor if weight_correction else 1.0
        s = w * self.__sink_poly(v / w)
        if include_airmass:
            s += self.__v_air_vert
        return s

    def sink_deriv(self, v):
        """
        Evaluate the derivative of the fitted sink-rate polynomial with respect to true airspeed, applying the instance weight factor.

        Parameters:
            v: True airspeed at which to evaluate the derivative (speed quantity).

        Returns:
            The derivative of sink rate with respect to true airspeed at `v`, with the configured weight scaling applied (sink change per unit speed).
        """
        w = self.__weight_factor
        return self.__sink_deriv_poly(v / w)

    # Average cross-country speed accounting for thermalling time
    # using Eq (2) from the included document "MacCready Speed to Fly Theory.pdf"
    def v_avg(self, v, mc):
        """
        Compute the net cross-country average speed accounting for thermalling time.

        Parameters:
            v (Quantity or float): True airspeed at which to evaluate average speed.
            mc (Quantity or float): MacCready setting (expected vertical speed).

        Returns:
            Vavg (Quantity or float): Net cross-country speed corresponding to the given airspeed and MacCready setting.
        """
        x = self.__v_air_horiz
        f = 1.0
        s = self.sink(v)
        return (mc * (v + x) - f * x * s) / (mc - s)

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
        """
        Compute the residual of the Reichmann speed-to-fly (STF) Equation II for root finding.

        Parameters:
            v (float): True airspeed in meters per second.
            mc (float): MacCready setting (climb rate) in meters per second.

        Returns:
            float: The residual s - v * s' - mc, where s is the glider sink rate (including ambient vertical air mass)
                   evaluated at v, and s' is its derivative with respect to airspeed. A root of this value corresponds
                   to the Reichmann STF solution.
        """
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

        return (s - ((1 - f) * ax + v) * s_deriv - mc) / (mc - s) ** 2

    # Fit the polar data to a polynomial
    # degree: the degree (order) of the polynomial to use
    def fit_polar(self, degree):
        """
        Fit a polynomial to the glider's sink-rate (polar) data and record fit diagnostics.

        Sets instance attributes for the fitted polynomial, its derivative, the fitted degree, and the fitted speed range. Appends R^2 and mean squared error (MSE) information to the instance message log. For degree == 2, speeds below the speed at minimum sink are excluded from the fit to avoid poor curvature near stall; for other degrees the full dataset is used.

        Parameters:
            degree (int): Polynomial degree to fit to the sink-rate data.
        """
        self.__degree = degree
        speed, sink = self.__glider.polar_data_magnitude()
        self.speed_range = (min(speed), max(speed))
        logger.debug(
            f"Fitting polar of degree {degree} over speed range {self.speed_range[0]:.3f} to {self.speed_range[1]:.3f} m/s"
        )

        # Low-order fits should ignore polar data at speeds below minimum sink
        # because the model cannot follow the curvature near stall speed
        if degree <= 3:
            min_sink_index = np.argmax(sink)
            start_index = min_sink_index
        else:
            start_index = 0

        self.__sink_poly, (SSE, _rank, _sv, _rcond) = Poly.fit(
            speed[start_index:], sink[start_index:], degree, full=True
        )
        logger.debug(f"Fitted polar polynomial coefficients: {self.__sink_poly.coef}")
        self.__sink_deriv_poly = self.__sink_poly.deriv()

        # Generate predicted y-values
        sink_predicted = self.__sink_poly(speed)

        # Calculate the R-value (Pearson correlation coefficient)
        r_value, _p_value = stats.pearsonr(
            sink[start_index:], sink_predicted[start_index:]
        )

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

        logger.info(f"{self.__messages}")

    def normal_solver(self, initial_guess, mc):
        """
        Find a root of the configured goal function using the provided initial speed guess.

        Parameters:
            initial_guess (float): Initial speed guess in meters per second.
            mc (pint.Quantity or float): MacCready setting passed to the goal function.

        Returns:
            Root speed in meters per second if a root is found, `None` otherwise.
        """
        [sol, _, err, _msg] = fsolve(
            self.goal_function, initial_guess, (mc), full_output=True, xtol=1.0e-6
        )
        if err == 1:
            solution = sol[0]
        else:
            # fsolve did not find a solution
            solution = None

        return solution

    def bruteforce_solver(self, search_range, mc):
        # First, find a workable range
        """
        Finds a root of the configured goal function for a given MacCready setting by locating a sign-changing bracket and solving it with Brent's method.

        Parameters:
            search_range (tuple): (lower, upper) speed bounds in m/s used to locate a sign change in the goal function.
            mc (float): MacCready setting in m/s passed to the goal function.

        Returns:
            float or None: The speed in m/s where the goal function is zero if a bracket is found and the solver converges; `None` if no sign change is located or the solver fails to converge.
        """
        working_range = search_range
        r0_val = self.goal_function(working_range[0], mc)
        r1_val = self.goal_function(working_range[1], mc)
        solution = None
        if r0_val * r1_val > 0:
            # goal function has same sign at both ends of range
            # Brute force search for a better range
            # logger.debug(
            #     f"Initial failure: ({working_range[0]:.2f}, {working_range[1]:.2f}) = ({r0_val:.6f}, {r1_val:.6f})"
            # )
            r0 = working_range[0]
            for r1 in np.arange(working_range[0], 1.5 * working_range[1], 5.0):
                r1_val = self.goal_function(r1, mc)
                if r0_val * r1_val < 0:
                    working_range = (r0, r1)
                    logger.debug(
                        f"New range: ({working_range[0]}, {working_range[1]}) = ({r0_val:.6f}, {r1_val:.6f})"
                    )
                    break
                r0 = r1
                r0_val = r1_val

        if r0_val * r1_val > 0:
            # self.__messages += f"\nNo sign change in goal function for MC = {mc:.3f} m/s over range {search_range[0]:0.2f} to {search_range[1]:0.2f} m/s\n"
            solution = None
        else:
            # bruteforce solver
            sol = root_scalar(
                self.goal_function,
                bracket=working_range,
                args=(mc,),
                method="brentq",
                xtol=1.0e-6,
            )
            solution = sol.root if sol.converged else None
        return solution

        """
        Find the speed-to-fly that optimizes performance for each value in a sequence of MacCready settings.
        Parameters:
            mc (float): MacCready setting (m/s).
        """

    # mcTable has MacCready values in units of m/s
    def MacCready(self, mcTable):
        """
        Compute a MacCready performance table for a sequence of MacCready settings.

        Parameters:
            mcTable (array-like of pint.Quantity): Sequence of MacCready values with units of meters per second.

        Returns:
            pandas.DataFrame: Table with columns
                - MC: MacCready settings as pint quantities in m/s
                - STF: Optimal speed-to-fly (STF) for each MC as pint quantities in m/s
                - Vavg: Net cross-country average speed (accounts for thermalling) as pint quantities in m/s
                - L/D: Lift-to-drag ratio at the STF (dimensionless)
                - solverResult: Value of the goal-function (solver residual) at the found STF
        """
        # optimum speed-to-fly
        Vstf = np.zeros(len(mcTable))

        # net cross-country speed, taking thermalling time into account
        Vavg = np.zeros(len(mcTable))

        # L/D ratio at Vstf
        LD = np.zeros(len(mcTable))

        solver_result = np.zeros(len(mcTable))

        # Guess 50 knots, but must express as m/s
        initial_guess = ureg("50.0 knots").to(ureg.mps).magnitude * self.__weight_factor

        # For each MC value, find the speed at which "goal_function" is equal to zero
        solver_range = (
            self.speed_range[0] * self.__weight_factor,
            self.speed_range[1] * self.__weight_factor,
        )

        for i in range(len(mcTable)):
            mc = mcTable[i]
            v = self.normal_solver(initial_guess, mc.magnitude)

            if v is None:
                logger.debug(
                    f"Normal solver failed for MC={mc:.3f~P}; trying bruteforce solver"
                )
                v = self.bruteforce_solver(solver_range, mc.magnitude)
            else:
                # Check that a solution > slowest speed was found
                if v < solver_range[0]:
                    logger.debug(
                        f"Normal solver returned out-of-range solution v={v:.3f} m/s for MC={mc:.3f~P}; trying bruteforce solver"
                    )
                    v = self.bruteforce_solver(solver_range, mc.magnitude)

            if v is None:
                Vstf[i] = float("nan")  # no solution found
                Vavg[i] = float("nan")
                LD[i] = float("nan")
                solver_result[i] = float("nan")
                self.__messages += f"No solution found for MC = {mc:.3f~P}\n"
                logger.debug(f"No solution found for MC={mc:.3f~P}")
            else:
                Vstf[i] = v
                Vavg[i] = self.v_avg(v, mc.magnitude)
                S = self.sink(v) - self.__v_air_vert
                LD[i] = v / (-S)
                solver_result[i] = self.goal_function(v, mc.magnitude)
                initial_guess = v

        df_mc = pd.DataFrame(
            {
                "MC": PA_(mcTable, ureg.mps),
                "STF": PA_(Vstf, ureg.mps),
                "Vavg": PA_(Vavg, ureg.mps),
                "L/D": LD,
                "solverResult": solver_result,
            }
        )
        return df_mc
