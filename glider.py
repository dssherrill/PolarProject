import logging
import numpy as np
import pandas as pd
import json

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import fsolve
from scipy import stats
import pint_pandas

# Get access to the one-and-only UnitsRegistry instance
from units import ureg

PA_ = pint_pandas.PintArray
logger = logging.getLogger(__name__)


class Glider:
    def __init__(self, current_glider):
        logger = logging.getLogger(__name__)
        logger.debug("Entering Glider.__init__")

        self.__messages = ""
        self.__external_poly = None
        self.__external_speed_range = None

        # Get units from the passed DataFrame
        speed_units = ureg(current_glider["polarSpeedUnits"].iloc[0])
        sink_units = ureg(current_glider["polarSinkUnits"].iloc[0])

        # Get glider data from the passed DataFrame
        self.__name = current_glider["name"].iloc[0]
        self.__ref_weight = ureg(current_glider["referenceWeight"].iloc[0]).to("kg")
        self.__empty_weight = ureg(current_glider["emptyWeight"].iloc[0]).to("kg")
        self.__wing_area = ureg(current_glider["wingArea"].iloc[0]).to("m**2")

        # Load the polar data: either from an external polynomial or a CSV file
        if "polarCoefficients" in current_glider.columns and isinstance(
            current_glider["polarCoefficients"].iloc[0], list
        ):
            self.load_polynomial(current_glider, speed_units, sink_units)
        else:
            self.load_CSV(current_glider, speed_units, sink_units)

        logger.debug("Exiting Glider.__init__")

    # Throughout this code, the conventional units are
    #  m/s for speed and
    #  kg for weight

    # Reads a polar from a CSV file created by WebPlotDigitizer at https://automeris.io/
    # x = first column  = speed in km per hour
    # y = second column = sink in m per second (all values should be negative)
    def load_CSV(self, current_glider, speed_units, sink_units):
        """
        Load polar speed and sink data from a CSV file referenced by the provided glider record and store them as pint-quantified NumPy arrays.

        Parameters:
            current_glider (pandas.DataFrame): A one-row DataFrame containing glider metadata; must include a 'polarFileName' column whose first value is the CSV filename located under ./datafiles/.
            speed_units (pint.Unit or pint.Quantity): Unit or quantity to apply to the CSV speed column before converting to meters per second.
            sink_units (pint.Unit or pint.Quantity): Unit or quantity to apply to the CSV sink column before converting to meters per second.

        Raises:
            FileNotFoundError: If the CSV file specified by 'polarFileName' cannot be found under ./datafiles/.
            RuntimeError: If an unexpected error occurs while reading the CSV file.
            ValueError: If the loaded speed or sink arrays are empty after reading the CSV.
        """

        polar_file_name = current_glider["polarFileName"].iloc[0]
        logger.debug(f'polarFileName is "{polar_file_name}"')

        file_path = f"./datafiles/{polar_file_name}"
        logger.debug(f'file_path is "{file_path}"')

        try:
            df_polar = pd.read_csv(file_path, header=None)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            raise FileNotFoundError(f"Polar data file not found: {file_path}") from None
        except Exception as e:
            self.__messages += (
                f"An unexpected error occurred ({type(e).__name__}): {e}\n"
            )
            raise RuntimeError(f"Failed to load polar data: {e}") from e

        # Convert speed data to m/s
        self.__speed_data = (df_polar.iloc[:, 0].to_numpy() * speed_units).to("m/s")

        # Converts sink data to m/s
        self.__sink_data = (df_polar.iloc[:, 1].to_numpy() * sink_units).to("m/s")

        # Ensure CSV data loaded successfully before fitting
        if len(self.__speed_data) == 0:
            raise ValueError("Polar speed data is empty; cannot fit polar.")
        if len(self.__sink_data) == 0:
            raise ValueError("Polar sink data is empty; cannot fit polar.")

    def load_polynomial(self, current_glider, speed_units, sink_units):
        """
        Construct the polar polynomial from coefficients provided in the glider data.

        Converts the polynomial from its native (speed_units, sink_units) domain to the
        internal (m/s, m/s) domain and stores the speed range for the polynomial.

        Parameters:
            current_glider (pandas.DataFrame): One-row DataFrame containing glider metadata;
                must include 'polarCoefficients' (list, lowest-to-highest order), 'minSpeed',
                and 'maxSpeed' columns.
            speed_units (pint.Unit): Units of the polynomial's speed input variable.
            sink_units (pint.Unit): Units of the polynomial's sink-rate output variable.
        """
        coefficients = current_glider["polarCoefficients"].iloc[0]

        # Convert polynomial from (speed_units, sink_units) domain to (m/s, m/s).
        # If p(x) is the polynomial with x in speed_units and output in sink_units, then
        # the converted polynomial q(v) with v in m/s satisfies q(v) = p(v / speed_to_ms) * sink_to_ms,
        # giving converted coefficient b_k = a_k * sink_to_ms / speed_to_ms**k.
        speed_to_ms = (1.0 * speed_units).to("m/s").magnitude
        sink_to_ms = (1.0 * sink_units).to("m/s").magnitude
        converted_coeffs = [
            a * sink_to_ms / (speed_to_ms**k) for k, a in enumerate(coefficients)
        ]
        self.__external_poly = Poly(converted_coeffs)

        min_speed = ureg(current_glider["minSpeed"].iloc[0]).to("m/s").magnitude
        max_speed = ureg(current_glider["maxSpeed"].iloc[0]).to("m/s").magnitude
        self.__external_speed_range = (min_speed, max_speed)

    def has_external_polynomial(self):
        """Return True if this glider's polar is defined by external polynomial coefficients."""
        return self.__external_poly is not None

    def external_polynomial(self):
        """
        Return the polar polynomial constructed from external coefficients, in (m/s, m/s) domain.

        Returns:
            numpy.polynomial.Polynomial or None
        """
        return self.__external_poly

    def external_speed_range(self):
        """
        Return the valid speed range (min, max) in m/s for the external polynomial polar.

        Returns:
            tuple(float, float) or None
        """
        return self.__external_speed_range

    def polar_data_magnitude(self):
        """
        Retrieve the loaded polar speed and sink datasets.

        Returns:
            tuple: (speed_data, sink_data)
                speed_data: Sequence of speeds as floats in meters per second.
                sink_data: Corresponding sink rates as floats in meters per second.
        """
        return self.__speed_data.magnitude, self.__sink_data.magnitude

    def get_speed_data(self):
        return self.__speed_data

    def get_sink_data(self):
        return self.__sink_data

    def reference_weight(self):
        """
        Return the stored reference weight of the glider.

        Returns:
            float: Reference weight in kilograms.
        """
        return self.__ref_weight

    def reference_pilot_weight(self):
        """
        Compute the glider's reference pilot weight.

        Reference pilot weight is the stored reference weight minus the stored empty weight.

        Returns:
            pilot_weight (float): Reference pilot weight in kilograms.
        """
        return self.__ref_weight - self.__empty_weight

    def reference_wing_loading(self):
        """
        Compute the glider's reference wing loading.

        Reference wing loading is the stored reference weight divided by the stored wing area.

        Returns:
            wing_loading (float): Reference wing loading in kilograms per square meter (kg/m^2).
        """
        return self.__ref_weight / self.__wing_area

    def wing_loading(self, pilot_weight):
        """
        Compute the wing loading for a given pilot weight.

        Parameters:
            pilot_weight (pint.Quantity): Pilot weight in kilograms.

        Returns:
            float or pint.Quantity: Wing loading in kilograms per square meter (kg/m^2)
        """
        if pilot_weight is None:
            return self.reference_wing_loading()
        else:
            return (self.__empty_weight + pilot_weight) / self.__wing_area

    def pilot_weight(self, wing_loading):
        """
        Compute the pilot weight corresponding to a given wing loading.

        Parameters:
            wing_loading (pint.Quantity): current wing loading in kilograms per square meter (kg/m^2).

        Returns:
            pint.Quantity: Pilot weight in kilograms
        """
        if wing_loading is None:
            return self.reference_pilot_weight()
        else:
            return (wing_loading * self.__wing_area) - self.__empty_weight

    def empty_weight(self):
        """
        Get the glider's stored empty weight.

        Returns:
            empty_weight (float): The empty weight of the glider in kilograms.
        """
        return self.__empty_weight

    def wing_area(self):
        return self.__wing_area

    def messages(self):
        """
        Return the accumulated status and error messages recorded by this Polar instance.

        Returns:
            str: Concatenated message string collected during operations (may be empty).
        """
        return self.__messages

    def name(self):
        return self.__name
