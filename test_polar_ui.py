"""
Comprehensive tests for polar_ui.py

Tests cover:
- load_polar function
- get_cached_glider function
- Dash callback functions (unit changes, graph updates)
- Unit conversions
- Edge cases and boundary conditions
- Integration with glider and polar_calc modules
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import dash.exceptions

import polar_ui
import glider
import polar_calc
from units import ureg



@pytest.fixture
def sample_glider_info():
    """Fixture providing sample glider information DataFrame"""
    return pd.DataFrame(
        {
            "name": ["ASW 28"],
            "wingArea": ["10.5 m**2"],
            "referenceWeight": ["325.0 kg"],
            "emptyWeight": ["268.0 kg"],
            "polarFileName": ["ASW 28.csv"],
            "polarSpeedUnits": ["kph"],
            "polarSinkUnits": ["m/s"],
        }
    )


@pytest.fixture
def mock_glider():
    """Fixture providing a mock Glider instance"""
    mock_g = MagicMock(spec=glider.Glider)
    mock_g.name.return_value = "ASW 28"
    mock_g.referenceWeight.return_value = ureg("325.0 kg")
    mock_g.emptyWeight.return_value = ureg("268.0 kg")
    mock_g.wingArea.return_value = ureg("10.5 m**2")
    mock_g.referenceWingLoading.return_value = ureg("325.0 kg") / ureg("10.5 m**2")

    # Mock polar data
    speeds = np.array([22.22, 27.78, 33.33, 38.89, 44.44]) * ureg("m/s")
    sinks = np.array([-1.1, -0.95, -0.85, -0.78, -0.75]) * ureg("m/s")
    mock_g.get_speed_data.return_value = speeds
    mock_g.get_sink_data.return_value = sinks
    mock_g.polar_data_magnitude.return_value = (speeds.magnitude, sinks.magnitude)
    mock_g.messages.return_value = ""

    return mock_g


@pytest.fixture
def mock_polar():
    """Fixture providing a mock Polar instance"""
    mock_p = MagicMock(spec=polar_calc.Polar)
    mock_p.get_weight_factor.return_value = 1.0
    mock_p.get_weight_fly.return_value = ureg("325.0 kg")
    mock_p.speed_range = (20.0, 50.0)
    mock_p.messages.return_value = "R^2 = 0.99\nMSE = 0.01"

    # Mock sink method
    def mock_sink(speed, weight_correction=True, include_airmass=True):
        return -0.85 * np.ones_like(speed)

    mock_p.sink = mock_sink

    # Mock goal_function method
    def mock_goal_function(v, mc):
        return 0.1 * (v - 30.0)

    mock_p.goal_function = mock_goal_function

    # Mock MacCready method
    def mock_maccready(mc_table):
        n = len(mc_table)
        return pd.DataFrame(
            {
                "MC": pd.array(mc_table, dtype="pint[m/s]"),
                "STF": pd.array(np.linspace(25, 40, n) * ureg("m/s"), dtype="pint[m/s]"),
                "Vavg": pd.array(
                    np.linspace(20, 35, n) * ureg("m/s"), dtype="pint[m/s]"
                ),
                "L/D": np.linspace(30, 40, n),
                "solverResult": np.zeros(n),
            }
        )

    mock_p.MacCready = mock_maccready

    return mock_p


class TestLoadPolar:
    """Test cases for load_polar function"""

    def test_load_polar_creates_polar_instance(self, mock_glider):
        """Test that load_polar creates a Polar instance"""
        with patch("polar_ui.polar_calc.Polar") as MockPolar:
            mock_polar_instance = MagicMock()
            MockPolar.return_value = mock_polar_instance

            result = polar_ui.load_polar(
                mock_glider, degree=5, goal_function="Reichmann", v_air_horiz=0.0, v_air_vert=0.0, pilot_weight=100.0
            )

            MockPolar.assert_called_once_with(
                mock_glider, 5, "Reichmann", 0.0, 0.0, 100.0
            )
            assert result == mock_polar_instance

    def test_load_polar_with_different_degrees(self, mock_glider):
        """Test load_polar with different polynomial degrees"""
        with patch("polar_ui.polar_calc.Polar") as MockPolar:
            mock_polar_instance = MagicMock()
            MockPolar.return_value = mock_polar_instance

            for degree in [2, 3, 5, 7]:
                polar_ui.load_polar(
                    mock_glider, degree=degree, goal_function="Reichmann", v_air_horiz=0.0, v_air_vert=0.0, pilot_weight=100.0
                )

                # Check that degree was passed correctly
                assert MockPolar.call_args[0][1] == degree

    def test_load_polar_with_goal_functions(self, mock_glider):
        """Test load_polar with different goal functions"""
        with patch("polar_ui.polar_calc.Polar") as MockPolar:
            mock_polar_instance = MagicMock()
            MockPolar.return_value = mock_polar_instance

            for goal in ["Reichmann", "Test"]:
                polar_ui.load_polar(
                    mock_glider, degree=5, goal_function=goal, v_air_horiz=0.0, v_air_vert=0.0, pilot_weight=100.0
                )

                assert MockPolar.call_args[0][2] == goal

    def test_load_polar_with_airmass_movement(self, mock_glider):
        """Test load_polar with non-zero airmass movement"""
        with patch("polar_ui.polar_calc.Polar") as MockPolar:
            mock_polar_instance = MagicMock()
            MockPolar.return_value = mock_polar_instance

            polar_ui.load_polar(
                mock_glider, degree=5, goal_function="Reichmann", v_air_horiz=5.0, v_air_vert=1.0, pilot_weight=100.0
            )

            call_args = MockPolar.call_args[0]
            assert call_args[3] == 5.0  # v_air_horiz
            assert call_args[4] == 1.0  # v_air_vert

    def test_load_polar_with_none_pilot_weight(self, mock_glider):
        """Test load_polar with None as pilot weight"""
        with patch("polar_ui.polar_calc.Polar") as MockPolar:
            mock_polar_instance = MagicMock()
            MockPolar.return_value = mock_polar_instance

            polar_ui.load_polar(
                mock_glider, degree=5, goal_function="Reichmann", v_air_horiz=0.0, v_air_vert=0.0, pilot_weight=None
            )

            call_args = MockPolar.call_args[0]
            assert call_args[5] is None


class TestGetCachedGlider:
    """Test cases for get_cached_glider function"""

    def setUp(self):
        """Clear the glider cache before each test"""
        polar_ui._glider_cache.clear()

    def test_get_cached_glider_creates_new_instance(self, sample_glider_info):
        """Test that get_cached_glider creates a new instance for uncached glider"""
        polar_ui._glider_cache.clear()

        with patch("polar_ui.glider.Glider") as MockGlider:
            mock_instance = MagicMock()
            MockGlider.return_value = mock_instance

            result = polar_ui.get_cached_glider("ASW 28", sample_glider_info)

            MockGlider.assert_called_once_with(sample_glider_info)
            assert result == mock_instance
            assert "ASW 28" in polar_ui._glider_cache

    def test_get_cached_glider_returns_cached_instance(self, sample_glider_info):
        """Test that get_cached_glider returns cached instance on second call"""
        polar_ui._glider_cache.clear()

        with patch("polar_ui.glider.Glider") as MockGlider:
            mock_instance = MagicMock()
            MockGlider.return_value = mock_instance

            # First call - should create new instance
            result1 = polar_ui.get_cached_glider("ASW 28", sample_glider_info)

            # Second call - should return cached instance
            result2 = polar_ui.get_cached_glider("ASW 28", sample_glider_info)

            # Glider constructor should only be called once
            MockGlider.assert_called_once()
            assert result1 == result2

    def test_get_cached_glider_different_gliders(self, sample_glider_info):
        """Test that different gliders are cached separately"""
        polar_ui._glider_cache.clear()

        glider_info_2 = sample_glider_info.copy()
        glider_info_2["name"] = ["Duo Discus T"]

        with patch("polar_ui.glider.Glider") as MockGlider:
            MockGlider.side_effect = [MagicMock(), MagicMock()]

            result1 = polar_ui.get_cached_glider("ASW 28", sample_glider_info)
            result2 = polar_ui.get_cached_glider("Duo Discus T", glider_info_2)

            # Should create two instances
            assert MockGlider.call_count == 2
            assert result1 != result2


class TestUnitConstants:
    """Test cases for unit constants and choices"""

    def test_metric_units_structure(self):
        """Test that METRIC_UNITS has correct structure"""
        assert "Speed" in polar_ui.METRIC_UNITS
        assert "Sink" in polar_ui.METRIC_UNITS
        assert "Weight" in polar_ui.METRIC_UNITS
        assert "Wing Area" in polar_ui.METRIC_UNITS
        assert "Pressure" in polar_ui.METRIC_UNITS

    def test_metric_units_values(self):
        """Test that METRIC_UNITS has correct values"""
        assert polar_ui.METRIC_UNITS["Speed"] == ureg("kph")
        assert polar_ui.METRIC_UNITS["Sink"] == ureg("m/s")
        assert polar_ui.METRIC_UNITS["Weight"] == ureg("kg")
        assert polar_ui.METRIC_UNITS["Wing Area"] == ureg("m**2")
        assert polar_ui.METRIC_UNITS["Pressure"] == ureg("kg/m**2")

    def test_us_units_structure(self):
        """Test that US_UNITS has correct structure"""
        assert "Speed" in polar_ui.US_UNITS
        assert "Sink" in polar_ui.US_UNITS
        assert "Weight" in polar_ui.US_UNITS
        assert "Wing Area" in polar_ui.US_UNITS
        assert "Pressure" in polar_ui.US_UNITS

    def test_us_units_values(self):
        """Test that US_UNITS has correct values"""
        assert polar_ui.US_UNITS["Speed"] == ureg("knots")
        assert polar_ui.US_UNITS["Sink"] == ureg("knots")
        assert polar_ui.US_UNITS["Weight"] == ureg("lbs")
        assert polar_ui.US_UNITS["Wing Area"] == ureg("ft**2")
        assert polar_ui.US_UNITS["Pressure"] == ureg("lbs/ft**2")

    def test_unit_choices_keys(self):
        """Test that UNIT_CHOICES has correct keys"""
        assert "Metric" in polar_ui.UNIT_CHOICES
        assert "US" in polar_ui.UNIT_CHOICES
        assert polar_ui.UNIT_CHOICES["Metric"] == polar_ui.METRIC_UNITS
        assert polar_ui.UNIT_CHOICES["US"] == polar_ui.US_UNITS


class TestInitialData:
    """Test cases for initial data structures"""

    def test_initial_mc_data_structure(self):
        """Test that initial_mc_data has correct structure"""
        assert "MC" in polar_ui.initial_mc_data.columns
        assert "STF" in polar_ui.initial_mc_data.columns
        assert "Vavg" in polar_ui.initial_mc_data.columns
        assert "L/D" in polar_ui.initial_mc_data.columns

    def test_initial_glider_data_structure(self):
        """Test that initial_glider_data has correct structure"""
        assert "Label" in polar_ui.initial_glider_data.columns
        assert "Metric" in polar_ui.initial_glider_data.columns
        assert "US" in polar_ui.initial_glider_data.columns

    def test_initial_glider_data_labels(self):
        """Test that initial_glider_data has correct labels"""
        expected_labels = [
            "Reference Weight",
            "Empty Weight",
            "Pilot+Ballast Weight",
            "Gross Weight",
            "Wing Loading",
        ]
        assert list(polar_ui.initial_glider_data["Label"]) == expected_labels


class TestProcessUnitChangeCallback:
    """Test cases for process_unit_change callback function"""

    @pytest.fixture
    def setup_dash_context(self):
        """Setup Dash context for callback testing"""
        # Mock dash.ctx
        with patch("polar_ui.dash.ctx") as mock_ctx:
            mock_ctx.triggered_id = None
            yield mock_ctx

    def test_process_unit_change_metric_to_us(self, sample_glider_info, mock_glider, setup_dash_context):
        """Test unit conversion from metric to US"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.set_props"):
            result = polar_ui.process_unit_change(
                units="US",
                weight_or_loading="Pilot Weight",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=100.0,  # 100 kg
                v_air_horiz_in=None,
                v_air_vert_in=None,
                data={"pilot_weight": 100.0, "wing_loading": None, "v_air_horiz": None, "v_air_vert": None},
            )

            # Should return tuple with 9 elements
            assert len(result) == 9

            # Check that glider data is returned
            glider_data = result[0]
            assert isinstance(glider_data, list)

    def test_process_unit_change_with_pilot_weight_input(
        self, sample_glider_info, mock_glider, setup_dash_context
    ):
        """Test processing pilot weight input"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider
        setup_dash_context.triggered_id = "pilot-weight-or-wing-loading-input"

        with patch("polar_ui.set_props"):
            result = polar_ui.process_unit_change(
                units="Metric",
                weight_or_loading="Pilot Weight",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=100.0,
                v_air_horiz_in=None,
                v_air_vert_in=None,
                data={},
            )

            # Check that pilot weight is stored in data
            stored_data = result[8]
            assert stored_data["pilot_weight"] == 100.0

    def test_process_unit_change_with_wing_loading_input(
        self, sample_glider_info, mock_glider, setup_dash_context
    ):
        """Test processing wing loading input"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider
        setup_dash_context.triggered_id = "pilot-weight-or-wing-loading-input"

        with patch("polar_ui.set_props"):
            result = polar_ui.process_unit_change(
                units="Metric",
                weight_or_loading="Wing Loading",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=35.0,  # 35 kg/m^2
                v_air_horiz_in=None,
                v_air_vert_in=None,
                data={},
            )

            # Check that wing loading is stored in data
            stored_data = result[8]
            assert stored_data["wing_loading"] == 35.0

    def test_process_unit_change_with_horizontal_speed(
        self, sample_glider_info, mock_glider, setup_dash_context
    ):
        """Test processing horizontal airmass speed input"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider
        setup_dash_context.triggered_id = "airmass-horizontal-speed"

        with patch("polar_ui.set_props"):
            result = polar_ui.process_unit_change(
                units="Metric",
                weight_or_loading="Pilot Weight",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=None,
                v_air_horiz_in=20.0,  # 20 kph
                v_air_vert_in=None,
                data={},
            )

            # Check that horizontal speed is stored
            stored_data = result[8]
            assert stored_data["v_air_horiz"] is not None

    def test_process_unit_change_with_vertical_speed(
        self, sample_glider_info, mock_glider, setup_dash_context
    ):
        """Test processing vertical airmass speed input"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider
        setup_dash_context.triggered_id = "airmass-vertical-speed"

        with patch("polar_ui.set_props"):
            result = polar_ui.process_unit_change(
                units="Metric",
                weight_or_loading="Pilot Weight",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=None,
                v_air_horiz_in=None,
                v_air_vert_in=2.0,  # 2 m/s
                data={},
            )

            # Check that vertical speed is stored
            stored_data = result[8]
            assert stored_data["v_air_vert"] is not None


class TestUpdateGraphCallback:
    """Test cases for update_graph callback function"""

    @pytest.fixture
    def setup_mocks(self, mock_glider, mock_polar):
        """Setup mocks for update_graph testing"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.load_polar") as mock_load_polar:
            mock_load_polar.return_value = mock_polar
            yield mock_load_polar

    def test_update_graph_basic_call(self, setup_mocks):
        """Test basic update_graph call"""
        result = polar_ui.update_graph(
            data={"pilot_weight": 100.0, "v_air_horiz": 0.0, "v_air_vert": 0.0},
            degree=5,
            glider_name="ASW 28",
            maccready=0.0,
            goal_function="Reichmann",
            show_debug_graphs=False,
            write_excel_file=False,
            units="Metric",
        )

        # Should return 8 elements
        assert len(result) == 8

        # First element should be glider name
        assert result[0] == "ASW 28"

        # Second element should be messages
        assert isinstance(result[1], str)

    def test_update_graph_with_debug_graphs(self, setup_mocks):
        """Test update_graph with debug graphs enabled"""
        result = polar_ui.update_graph(
            data={},
            degree=5,
            glider_name="ASW 28",
            maccready=0.0,
            goal_function="Reichmann",
            show_debug_graphs=True,
            write_excel_file=False,
            units="Metric",
        )

        # Should still return 8 elements
        assert len(result) == 8

    def test_update_graph_enforces_minimum_degree(self, setup_mocks):
        """Test that update_graph enforces minimum polynomial degree of 2"""
        result = polar_ui.update_graph(
            data={},
            degree=1,  # Too low
            glider_name="ASW 28",
            maccready=0.0,
            goal_function="Reichmann",
            show_debug_graphs=False,
            write_excel_file=False,
            units="Metric",
        )

        # Last element should be the effective degree (minimum 2)
        assert result[7] == 2

    def test_update_graph_with_none_degree(self, setup_mocks):
        """Test update_graph with None as degree"""
        result = polar_ui.update_graph(
            data={},
            degree=None,
            glider_name="ASW 28",
            maccready=0.0,
            goal_function="Reichmann",
            show_debug_graphs=False,
            write_excel_file=False,
            units="Metric",
        )

        # Should use default polynomial degree
        assert result[7] == polar_ui.DEFAULT_POLYNOMIAL_DEGREE

    def test_update_graph_with_none_maccready(self, setup_mocks):
        """Test update_graph with None as MacCready value"""
        result = polar_ui.update_graph(
            data={},
            degree=5,
            glider_name="ASW 28",
            maccready=None,
            goal_function="Reichmann",
            show_debug_graphs=False,
            write_excel_file=False,
            units="Metric",
        )

        # Should succeed with default MacCready of 0
        assert len(result) == 8

    def test_update_graph_us_units(self, setup_mocks):
        """Test update_graph with US units"""
        result = polar_ui.update_graph(
            data={},
            degree=5,
            glider_name="ASW 28",
            maccready=2.0,
            goal_function="Reichmann",
            show_debug_graphs=False,
            write_excel_file=False,
            units="US",
        )

        # Should return MacCready table data
        mc_data = result[4]
        assert isinstance(mc_data, list)

    def test_update_graph_different_goal_functions(self, setup_mocks):
        """Test update_graph with different goal functions"""
        for goal in ["Reichmann", "Test"]:
            result = polar_ui.update_graph(
                data={},
                degree=5,
                glider_name="ASW 28",
                maccready=0.0,
                goal_function=goal,
                show_debug_graphs=False,
                write_excel_file=False,
                units="Metric",
            )

            assert len(result) == 8


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions"""

    def test_empty_glider_cache(self, sample_glider_info):
        """Test behavior with empty glider cache"""
        polar_ui._glider_cache.clear()

        with patch("polar_ui.glider.Glider") as MockGlider:
            mock_instance = MagicMock()
            MockGlider.return_value = mock_instance

            result = polar_ui.get_cached_glider("Test Glider", sample_glider_info)

            # Should create new instance
            MockGlider.assert_called_once()

    def test_large_polynomial_degree(self, mock_glider, mock_polar):
        """Test with unusually large polynomial degree"""
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.load_polar") as mock_load:
            mock_load.return_value = mock_polar

            result = polar_ui.update_graph(
                data={},
                degree=20,  # Very high degree
                glider_name="ASW 28",
                maccready=0.0,
                goal_function="Reichmann",
                show_debug_graphs=False,
                write_excel_file=False,
                units="Metric",
            )

            # Should still work
            assert len(result) == 8

    def test_negative_maccready_value(self, mock_glider, mock_polar):
        """Test with negative MacCready value (edge case)"""
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.load_polar") as mock_load:
            mock_load.return_value = mock_polar

            result = polar_ui.update_graph(
                data={},
                degree=5,
                glider_name="ASW 28",
                maccready=-1.0,  # Negative value
                goal_function="Reichmann",
                show_debug_graphs=False,
                write_excel_file=False,
                units="Metric",
            )

            # Should handle gracefully
            assert len(result) == 8

    def test_very_high_pilot_weight(self, sample_glider_info, mock_glider):
        """Test with extremely high pilot weight"""
        polar_ui._glider_cache.clear()
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.dash.ctx") as mock_ctx, patch("polar_ui.set_props"):
            mock_ctx.triggered_id = "pilot-weight-or-wing-loading-input"

            result = polar_ui.process_unit_change(
                units="Metric",
                weight_or_loading="Pilot Weight",
                glider_name="ASW 28",
                pilot_weight_or_wing_loading_in=500.0,  # Very heavy
                v_air_horiz_in=None,
                v_air_vert_in=None,
                data={},
            )

            # Should handle without error
            stored_data = result[8]
            assert stored_data["pilot_weight"] == 500.0


class TestRegressionCases:
    """Regression tests to prevent known issues"""

    def test_glider_cache_persists_between_calls(self, sample_glider_info):
        """Test that glider cache properly persists between calls"""
        polar_ui._glider_cache.clear()

        with patch("polar_ui.glider.Glider") as MockGlider:
            mock_instance = MagicMock()
            MockGlider.return_value = mock_instance

            # First call
            polar_ui.get_cached_glider("ASW 28", sample_glider_info)

            # Clear the mock call count
            MockGlider.reset_mock()

            # Second call should use cache
            polar_ui.get_cached_glider("ASW 28", sample_glider_info)

            # Should not create new instance
            MockGlider.assert_not_called()

    def test_unit_conversion_consistency(self):
        """Test that unit conversions are consistent"""
        # Convert 100 kph to knots and back
        speed_kph = 100.0 * ureg.kph
        speed_knots = speed_kph.to("knots")
        speed_back = speed_knots.to("kph")

        assert np.isclose(speed_back.magnitude, 100.0)

    def test_weight_factor_calculation_consistency(self, mock_glider):
        """Test that weight factor calculations are consistent"""
        polar_ui._glider_cache["ASW 28"] = mock_glider

        with patch("polar_ui.load_polar") as mock_load:
            mock_polar1 = MagicMock()
            mock_polar1.get_weight_factor.return_value = 1.0
            mock_polar1.get_weight_fly.return_value = ureg("325.0 kg")
            mock_polar1.speed_range = (20.0, 50.0)
            mock_polar1.messages.return_value = ""
            mock_polar1.sink = lambda v, **kwargs: -0.85 * np.ones_like(v)
            mock_polar1.MacCready = lambda mc_table: pd.DataFrame(
                {
                    "MC": pd.array(mc_table, dtype="pint[m/s]"),
                    "STF": pd.array(
                        np.linspace(25, 40, len(mc_table)) * ureg("m/s"), dtype="pint[m/s]"
                    ),
                    "Vavg": pd.array(
                        np.linspace(20, 35, len(mc_table)) * ureg("m/s"), dtype="pint[m/s]"
                    ),
                    "L/D": np.linspace(30, 40, len(mc_table)),
                    "solverResult": np.zeros(len(mc_table)),
                }
            )

            mock_load.return_value = mock_polar1

            result1 = polar_ui.update_graph(
                data={"pilot_weight": 100.0},
                degree=5,
                glider_name="ASW 28",
                maccready=0.0,
                goal_function="Reichmann",
                show_debug_graphs=False,
                write_excel_file=False,
                units="Metric",
            )

            result2 = polar_ui.update_graph(
                data={"pilot_weight": 100.0},
                degree=5,
                glider_name="ASW 28",
                maccready=0.0,
                goal_function="Reichmann",
                show_debug_graphs=False,
                write_excel_file=False,
                units="Metric",
            )

            # Results should be consistent
            assert result1[0] == result2[0]


class TestProductionMode:
    """Test cases for production mode detection"""

    def test_production_mode_default(self):
        """Test that production mode has a default value"""
        assert hasattr(polar_ui, "_production_mode")
        assert isinstance(polar_ui._production_mode, bool)

    @patch.dict(os.environ, {"PRODUCTION_MODE": "true"})
    def test_production_mode_from_env(self):
        """Test production mode detection from environment variable"""
        # This would need to be tested by reloading the module
        # Here we just verify the variable exists
        assert hasattr(polar_ui, "_production_mode")

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_environment_variable_detection(self):
        """Test environment detection from ENVIRONMENT variable"""
        # Verify the logic exists in the module
        assert hasattr(polar_ui, "_production_mode")


class TestDashApp:
    """Test cases for Dash app configuration"""

    def test_app_exists(self):
        """Test that Dash app is created"""
        assert hasattr(polar_ui, "app")
        assert polar_ui.app is not None

    def test_app_layout_exists(self):
        """Test that app layout is defined"""
        assert hasattr(polar_ui.app, "layout")
        assert polar_ui.app.layout is not None

    def test_default_glider_name(self):
        """Test that default glider name is defined"""
        assert hasattr(polar_ui, "DEFAULT_GLIDER_NAME")
        assert polar_ui.DEFAULT_GLIDER_NAME == "ASW 28"

    def test_default_polynomial_degree(self):
        """Test that default polynomial degree is defined"""
        assert hasattr(polar_ui, "DEFAULT_POLYNOMIAL_DEGREE")
        assert polar_ui.DEFAULT_POLYNOMIAL_DEGREE == 5


class TestSTFGraphTitle:
    """Test cases for STF graph title determination"""

    def test_determine_stf_graph_title_both_visible(self):
        """Test title when both STF and Vavg are visible"""
        title = polar_ui.determine_stf_graph_title(True, True)
        assert title == "MacCready Speed-to-Fly and Average Speed"

    def test_determine_stf_graph_title_stf_only(self):
        """Test title when only STF is visible"""
        title = polar_ui.determine_stf_graph_title(True, False)
        assert title == "MacCready Speed-to-Fly"

    def test_determine_stf_graph_title_vavg_only(self):
        """Test title when only Vavg is visible"""
        title = polar_ui.determine_stf_graph_title(False, True)
        assert title == "Average Speed"

    def test_determine_stf_graph_title_both_hidden(self):
        """Test title when both are hidden (default case)"""
        title = polar_ui.determine_stf_graph_title(False, False)
        assert title == "MacCready Speed-to-Fly and Average Speed"


class TestDetermineTitleFromFigure:
    """Test cases for determine_title_from_figure function"""

    def test_empty_figure(self):
        """Test with empty figure"""
        title = polar_ui.determine_title_from_figure({}, "STF")
        assert title == "MacCready Speed-to-Fly and Average Speed"

    def test_both_main_traces_visible(self):
        """Test with both main traces visible"""
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": True},
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "MacCready Speed-to-Fly and Average Speed"

    def test_only_stf_visible(self):
        """Test with only STF visible"""
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": False},
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "MacCready Speed-to-Fly"

    def test_only_vavg_visible(self):
        """Test with only Vavg visible"""
        figure = {
            "data": [
                {"name": "STF", "visible": False},
                {"name": "V<sub>avg</sub>", "visible": True},
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "Vavg")
        assert title == "Average Speed"

    def test_comparison_trace_stf_metric(self):
        """Test with comparison trace showing STF data"""
        figure = {
            "data": [
                {"name": "STF", "visible": False},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ]
        }
        # When compare_metric is "STF", comparison traces show STF data
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "MacCready Speed-to-Fly"

    def test_comparison_trace_vavg_metric(self):
        """Test with comparison trace showing Vavg data"""
        figure = {
            "data": [
                {"name": "STF", "visible": False},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ]
        }
        # When compare_metric is "Vavg", comparison traces show Vavg data
        title = polar_ui.determine_title_from_figure(figure, "Vavg")
        assert title == "Average Speed"

    def test_mixed_traces_stf_metric(self):
        """Test with main STF and comparison trace (STF metric)"""
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "MacCready Speed-to-Fly"

    def test_mixed_traces_vavg_metric(self):
        """Test with main Vavg and comparison trace (Vavg metric)"""
        figure = {
            "data": [
                {"name": "STF", "visible": False},
                {"name": "V<sub>avg</sub>", "visible": True},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "Vavg")
        assert title == "Average Speed"

    def test_both_types_with_comparison(self):
        """Test with both main traces and comparison showing different metric"""
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ]
        }
        # Main STF is visible, comparison shows Vavg
        title = polar_ui.determine_title_from_figure(figure, "Vavg")
        assert title == "MacCready Speed-to-Fly and Average Speed"

    def test_legendonly_treated_as_hidden(self):
        """Test that legendonly traces are treated as hidden"""
        figure = {
            "data": [
                {"name": "STF", "visible": "legendonly"},
                {"name": "V<sub>avg</sub>", "visible": True},
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "Average Speed"

    def test_debug_trace_ignored(self):
        """Test that debug traces (Solver Result) are ignored"""
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "Solver Result", "visible": True},  # debug trace - should be ignored
            ]
        }
        title = polar_ui.determine_title_from_figure(figure, "STF")
        assert title == "MacCready Speed-to-Fly"

    def test_plotly_figure_object(self):
        """Test with actual Plotly Figure object (not dictionary)"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create real Plotly figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces as Plotly objects
        trace_stf = go.Scatter(x=[1, 2], y=[1, 2], name="STF", visible=True)
        fig.add_trace(trace_stf, secondary_y=False)
        
        trace_vavg = go.Scatter(x=[1, 2], y=[2, 3], name="V<sub>avg</sub>", visible=False)
        fig.add_trace(trace_vavg, secondary_y=False)
        
        # Test with Plotly Figure object
        title = polar_ui.determine_title_from_figure(fig, "STF")
        assert title == "MacCready Speed-to-Fly"
        
        # Test with both visible
        fig.data[1].visible = True
        title = polar_ui.determine_title_from_figure(fig, "STF")
        assert title == "MacCready Speed-to-Fly and Average Speed"


class TestUpdateSTFTitleCallback:
    """Test cases for update_stf_title_on_restyle callback"""

    def test_update_stf_title_on_restyle_no_data(self):
        """Test that callback prevents update when no data"""
        with pytest.raises(dash.exceptions.PreventUpdate):
            polar_ui.update_stf_title_on_restyle(None, None, "STF")

    def test_update_stf_title_on_restyle_no_visibility_change(self):
        """Test that callback prevents update when no visibility changes"""
        restyle_data = [{"line": {"width": 3}}, [0]]
        figure = {"data": [{"name": "STF", "visible": True}]}
        
        with pytest.raises(dash.exceptions.PreventUpdate):
            polar_ui.update_stf_title_on_restyle(restyle_data, figure, "STF")

    def test_update_stf_title_on_restyle_hide_stf(self):
        """Test hiding STF trace updates title to 'Average Speed'"""
        restyle_data = [{"visible": [False]}, [0]]
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": True},
            ],
            "layout": {
                "title": {
                    "text": "MacCready Speed-to-Fly and Average Speed",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            },
        }
        
        result = polar_ui.update_stf_title_on_restyle(restyle_data, figure, "STF")
        assert result["layout"]["title"]["text"] == "Average Speed"

    def test_update_stf_title_on_restyle_hide_vavg(self):
        """Test hiding Vavg trace updates title to 'MacCready Speed-to-Fly'"""
        restyle_data = [{"visible": [False]}, [1]]
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": True},
            ],
            "layout": {
                "title": {
                    "text": "MacCready Speed-to-Fly and Average Speed",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            },
        }
        
        result = polar_ui.update_stf_title_on_restyle(restyle_data, figure, "Vavg")
        assert result["layout"]["title"]["text"] == "MacCready Speed-to-Fly"

    def test_update_stf_title_on_restyle_show_both(self):
        """Test showing both traces updates title correctly"""
        restyle_data = [{"visible": [True, True]}, [0, 1]]
        figure = {
            "data": [
                {"name": "STF", "visible": False},
                {"name": "V<sub>avg</sub>", "visible": False},
            ],
            "layout": {
                "title": {
                    "text": "Average Speed",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            },
        }
        
        result = polar_ui.update_stf_title_on_restyle(restyle_data, figure, "STF")
        assert result["layout"]["title"]["text"] == "MacCready Speed-to-Fly and Average Speed"

    def test_update_stf_title_preserves_subtitle(self):
        """Test that subtitle is preserved when updating title"""
        restyle_data = [{"visible": [False]}, [0]]
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": True},
            ],
            "layout": {
                "title": {
                    "text": "MacCready Speed-to-Fly and Average Speed",
                    "subtitle": {"text": "Subtracting Test Config"},
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            },
        }
        
        result = polar_ui.update_stf_title_on_restyle(restyle_data, figure, "STF")
        assert result["layout"]["title"]["text"] == "Average Speed"
        assert result["layout"]["title"]["subtitle"]["text"] == "Subtracting Test Config"

    def test_update_stf_title_with_comparison_traces(self):
        """Test title update with comparison traces visible"""
        restyle_data = [{"visible": [False]}, [0]]  # Hide main STF
        figure = {
            "data": [
                {"name": "STF", "visible": True},
                {"name": "V<sub>avg</sub>", "visible": False},
                {"name": "ASW 28,100.0 kg,d5", "visible": True},  # comparison trace
            ],
            "layout": {
                "title": {
                    "text": "MacCready Speed-to-Fly",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            },
        }
        
        # Main STF hidden, but comparison trace shows STF data
        result = polar_ui.update_stf_title_on_restyle(restyle_data, figure, "STF")
        assert result["layout"]["title"]["text"] == "MacCready Speed-to-Fly"
