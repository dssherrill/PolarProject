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

        # Get units from the passed DataFrame
        speed_units = ureg(current_glider["polarSpeedUnits"].iloc[0])
        sink_units = ureg(current_glider["polarSinkUnits"].iloc[0])

        # Get glider data from the passed DataFrame
        self.__name = current_glider["name"].iloc[0]
        self.__ref_weight = ureg(current_glider["referenceWeight"].iloc[0]).to("kg")
        self.__empty_weight = ureg(current_glider["emptyWeight"].iloc[0]).to("kg")
        self.__wing_area = ureg(current_glider["wingArea"].iloc[0]).to("m**2")

        # Load the polar data from the CSV file
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

    def referenceWeight(self):
        """
        Return the stored reference weight of the glider.
        
        Returns:
            float: Reference weight in kilograms.
        """
        return self.__ref_weight

    def referenceWingLoading(self):
        """
        Compute the glider's reference wing loading.
        
        Reference wing loading is the stored reference weight divided by the stored wing area.
        
        Returns:
            wing_loading (float): Reference wing loading in kilograms per square meter (kg/m^2).
        """
        return self.__ref_weight / self.__wing_area

    def emptyWeight(self):
        """
        Get the glider's stored empty weight.
        
        Returns:
            empty_weight (float): The empty weight of the glider in kilograms.
        """
        return self.__empty_weight

    def wingArea(self):
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