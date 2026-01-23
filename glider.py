import logging
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
logger = logging.getLogger(__name__)

class Glider:
    def __init__(self, current_glider):
        """
        Initialize a Glider instance from a DataFrame containing glider metadata and a polar file reference.
        
        Parameters:
            current_glider (pandas.DataFrame): A single-row DataFrame providing glider configuration. Required columns:
                - 'speedUnits', 'sinkUnits', 'weightUnits': unit strings understood by the module's unit registry.
                - 'name': glider name.
                - 'referenceWeight', 'emptyWeight': numeric weights expressed in the specified weight units.
                - 'polarFileName': filename of the CSV containing polar data (used by load_CSV).
        
        Side effects:
            - Sets internal attributes:
                - self.__messages (str): accumulator for status/error messages.
                - self.__name (str): glider name.
                - self.__ref_weight (pint.Quantity): reference weight converted to kilograms.
                - self.__empty_weight (pint.Quantity): empty weight converted to kilograms.
                - Loads and stores polar data by calling self.load_CSV(current_glider, speed_units, sink_units).
        """
        logger = logging.getLogger(__name__)
        logger.debug('Entering Glider.__init__')

        self.__messages = ""

        # Get units from the passed DataFrame
        speed_units = ureg(current_glider['speedUnits'].iloc[0])
        sink_units = ureg(current_glider['sinkUnits'].iloc[0])  
        weight_units = ureg(current_glider['weightUnits'].iloc[0])

        # Get glider data from the passed DataFrame
        self.__name = current_glider['name'].iloc[0]
        self.__ref_weight = (current_glider['referenceWeight'].iloc[0] * weight_units).to('kg')
        self.__empty_weight = (current_glider['emptyWeight'].iloc[0] * weight_units).to('kg')

        # Load the polar data from the CSV file
        self.load_CSV(current_glider, speed_units, sink_units)

        logger.debug('Exiting Glider.__init__')

    # Throughout this code, the conventional units are
    #  m/s for speed and 
    #  kg for weight

    # Reads a polar from a CSV file created by WebPlotDigitizer at https://automeris.io/
    # x = first column  = speed in km per hour
    # y = second column = sink in m per second (all values should be negative)
    def load_CSV(self, current_glider, speed_units, sink_units):
        """
        Load polar speed and sink data from the CSV named in `current_glider` and store them as pint-quantified arrays in meters per second.
        
        Parameters:
            current_glider (pandas.DataFrame): DataFrame containing a `polarFileName` column; the first row's value is used to locate the CSV under `./datafiles/`.
            speed_units (pint.Unit or pint.Quantity): Unit to apply to the first CSV column (speed).
            sink_units (pint.Unit or pint.Quantity): Unit to apply to the second CSV column (sink rate).
        
        Behavior:
            - Reads ./datafiles/{polarFileName} with no header; expects column 0 = speed, column 1 = sink.
            - Populates `self.__speed_data` and `self.__sink_data` with the CSV columns converted to meters per second and kept as pint-quantified NumPy arrays.
            - Appends a descriptive message to `self.__messages` and raises `FileNotFoundError` when the file is missing.
            - Appends a descriptive message to `self.__messages` and raises `RuntimeError` for other I/O/parsing errors.
            - Raises `ValueError` if either loaded column is empty after conversion.
        """

        polar_file_name = current_glider['polarFileName'].iloc[0]
        logger.debug(f'polarFileName is "{polar_file_name}"')

        file_path = f"./datafiles/{polar_file_name}"
        logger.debug(f'file_path is "{file_path}"')
        
        try:
            df_polar = pd.read_csv(file_path, header=None)
        except FileNotFoundError:
            self.__messages += f"Error: The file '{file_path}' was not found.\n"
            raise FileNotFoundError(f"Polar data file not found: {file_path}")
        except Exception as e:
            self.__messages += f"An unexpected error occurred ({type(e).__name__}): {e}\n"
            raise RuntimeError(f"Failed to load polar data: {e}") from e

        # Convert speed data to m/s
        self.__speed_data = (df_polar.iloc[:,0].to_numpy() * speed_units).to('m/s')

        # Converts sink data to m/s
        self.__sink_data = (df_polar.iloc[:,1].to_numpy() * sink_units).to('m/s')

        # Ensure CSV data loaded successfully before fitting
        if len(self.__speed_data) == 0:
            raise ValueError('Polar speed data is empty; cannot fit polar.')
        if len(self.__sink_data) == 0:
            raise ValueError('Polar sink data is empty; cannot fit polar.')


    def polar_data_magnitude(self):
        """
        Return the magnitudes of the loaded polar speed and sink datasets in meters per second.
        
        Returns:
            tuple: (speed_magnitudes, sink_magnitudes)
                speed_magnitudes: Sequence[float] — speeds in meters per second.
                sink_magnitudes: Sequence[float] — sink rates in meters per second.
        """
        return self.__speed_data.magnitude, self.__sink_data.magnitude
    
    def get_speed_data(self):
        """
        Retrieve the glider's speed data as a Pint-quantified array expressed in meters per second.
        
        Returns:
            speed_data (pint.Quantity or pint_pandas.PintArray): Array of speed values with units of meters per second.
        """
        return self.__speed_data
    
    def get_sink_data(self):
        """
        Return the glider's sink-rate dataset as a pint-quantified array in meters per second.
        
        Returns:
            sink_data (pint.Quantity or pint_pandas.PintArray): Array of sink rates converted to meters per second.
        """
        return self.__sink_data

    def referenceWeight(self):
        """
        Get the glider's reference weight.
        
        Returns:
            reference_weight (pint.Quantity): Reference weight expressed in kilograms.
        """
        return self.__ref_weight
    
    def emptyWeight(self):
        """
        Return the glider's empty weight.
        
        Returns:
            pint.quantity: Empty weight expressed in kilograms.
        """
        return self.__empty_weight
    
    def messages(self):
        """
        Get accumulated status and error messages recorded by the Glider.
        
        Returns:
            str: Concatenated messages collected during operations; empty string if none.
        """
        return self.__messages

    def name(self):
        """
        Retrieve the glider's name.
        
        Returns:
            str: The glider name from the input metadata.
        """
        return self.__name    