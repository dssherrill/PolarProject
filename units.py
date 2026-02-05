# units.py
from pint import UnitRegistry, set_application_registry

# There can only be a single UnitRegistry instance
ureg = UnitRegistry()

# Set as the application registry for pickling support
set_application_registry(ureg)
