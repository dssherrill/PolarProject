# units.py
from pint import UnitRegistry

# There can only be a single UnitRegistry instance
ureg = UnitRegistry()

# Optional: set as the application registry for pickling support
# from pint import set_application_registry
# set_application_registry(ureg)
