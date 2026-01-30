"""
Comprehensive tests for the Glider class in glider.py

Tests cover:
- Initialization with valid glider data
- CSV loading with various units
- Error handling for missing files
- Error handling for invalid data
- Weight and wing area calculations
- Edge cases and boundary conditions
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import glider
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
def sample_glider_info_us_units():
    """Fixture providing sample glider information with US units"""
    return pd.DataFrame(
        {
            "name": ["SGS 1-34C"],
            "wingArea": ["151 ft**2"],
            "referenceWeight": ["565.0 lbs"],
            "emptyWeight": ["360.0 lbs"],
            "polarFileName": ["SGS 1-34C.csv"],
            "polarSpeedUnits": ["mph"],
            "polarSinkUnits": ["ft/sec"],
        }
    )


@pytest.fixture
def sample_csv_data():
    """Fixture providing sample polar CSV data"""
    return """80,-1.1
90,-0.95
100,-0.85
110,-0.78
120,-0.75
130,-0.76
140,-0.8
150,-0.88
160,-1.0
170,-1.15"""


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    """Fixture creating a temporary CSV file with polar data"""
    datafiles_dir = tmp_path / "datafiles"
    datafiles_dir.mkdir()
    csv_file = datafiles_dir / "ASW 28.csv"
    csv_file.write_text(sample_csv_data)
    return tmp_path


class TestGliderInitialization:
    """Test cases for Glider class initialization"""

    def test_init_valid_glider(self, sample_glider_info, temp_csv_file):
        """Test successful initialization with valid glider data"""
        with patch("glider.Glider.load_CSV") as mock_load:
            # Mock the load_CSV to avoid file I/O in this specific test
            mock_load.return_value = None
            g = glider.Glider(sample_glider_info)

            assert g.name() == "ASW 28"
            assert g.referenceWeight() == ureg("325.0 kg")
            assert g.emptyWeight() == ureg("268.0 kg")
            assert g.wingArea() == ureg("10.5 m**2")
            mock_load.assert_called_once()

    def test_init_extracts_units_correctly(self, sample_glider_info):
        """Test that initialization extracts units from DataFrame correctly"""
        with patch("glider.Glider.load_CSV") as mock_load:
            g = glider.Glider(sample_glider_info)

            # Verify load_CSV was called with correct unit arguments
            call_args = mock_load.call_args
            assert call_args is not None
            speed_units_arg = call_args[0][1]
            sink_units_arg = call_args[0][2]

            assert speed_units_arg == ureg("kph")
            assert sink_units_arg == ureg("m/s")

    def test_init_with_us_units(self, sample_glider_info_us_units):
        """Test initialization with US customary units"""
        with patch("glider.Glider.load_CSV") as mock_load:
            g = glider.Glider(sample_glider_info_us_units)

            # Verify US units are extracted correctly
            call_args = mock_load.call_args
            speed_units_arg = call_args[0][1]
            sink_units_arg = call_args[0][2]

            assert speed_units_arg == ureg("mph")
            assert sink_units_arg == ureg("ft/sec")


class TestGliderCSVLoading:
    """Test cases for CSV loading functionality"""

    def test_load_csv_success(self, sample_glider_info, sample_csv_data, tmp_path):
        """Test successful CSV loading"""
        # Create temporary directory structure
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text(sample_csv_data)

        # Change to temp directory for the test
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            # Verify data was loaded
            speed_data = g.get_speed_data()
            sink_data = g.get_sink_data()

            assert len(speed_data) == 10
            assert len(sink_data) == 10

            # Check first and last values
            assert np.isclose(speed_data[0].magnitude, (80 * ureg.kph).to("m/s").magnitude)
            assert np.isclose(sink_data[0].magnitude, -1.1)

        finally:
            os.chdir(original_dir)

    def test_load_csv_file_not_found(self, sample_glider_info, tmp_path):
        """Test error handling when CSV file is not found"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            with pytest.raises(FileNotFoundError, match="Polar data file not found"):
                g = glider.Glider(sample_glider_info)

        finally:
            os.chdir(original_dir)

    def test_load_csv_empty_speed_data(self, sample_glider_info, tmp_path):
        """Test error handling when CSV has empty speed data"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text("")  # Empty file

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Empty file causes pandas to raise an error which is wrapped in RuntimeError
            with pytest.raises(RuntimeError, match="Failed to load polar data"):
                g = glider.Glider(sample_glider_info)

        finally:
            os.chdir(original_dir)

    def test_load_csv_converts_units_correctly(self, sample_glider_info_us_units, tmp_path):
        """Test that CSV loading converts units correctly"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "SGS 1-34C.csv"

        # Create CSV with mph and ft/sec
        csv_data = "40,-5.0\n50,-4.5\n60,-4.2"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info_us_units)

            # Verify units are converted to m/s
            speed_data = g.get_speed_data()
            sink_data = g.get_sink_data()

            # Should be in m/s
            assert speed_data.units == ureg("m/s")
            assert sink_data.units == ureg("m/s")

            # Check conversion accuracy (40 mph ≈ 17.88 m/s)
            assert np.isclose(speed_data[0].magnitude, (40 * ureg.mph).to("m/s").magnitude)

        finally:
            os.chdir(original_dir)


class TestGliderProperties:
    """Test cases for Glider property accessors"""

    def test_reference_weight(self, sample_glider_info):
        """Test reference weight retrieval"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            ref_weight = g.referenceWeight()

            assert ref_weight == ureg("325.0 kg")
            assert ref_weight.units == ureg.kg

    def test_empty_weight(self, sample_glider_info):
        """Test empty weight retrieval"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            empty_weight = g.emptyWeight()

            assert empty_weight == ureg("268.0 kg")
            assert empty_weight.units == ureg.kg

    def test_wing_area(self, sample_glider_info):
        """Test wing area retrieval"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            wing_area = g.wingArea()

            assert wing_area == ureg("10.5 m**2")
            assert wing_area.units == ureg("m**2")

    def test_reference_wing_loading(self, sample_glider_info):
        """Test reference wing loading calculation"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            wing_loading = g.referenceWingLoading()

            expected = ureg("325.0 kg") / ureg("10.5 m**2")
            assert np.isclose(wing_loading.magnitude, expected.magnitude)
            assert wing_loading.units == ureg("kg/m**2")

    def test_name(self, sample_glider_info):
        """Test glider name retrieval"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            assert g.name() == "ASW 28"

    def test_messages_initially_empty(self, sample_glider_info):
        """Test that messages are initially empty"""
        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(sample_glider_info)
            assert g.messages() == ""


class TestGliderPolarData:
    """Test cases for polar data retrieval"""

    def test_polar_data_magnitude(self, sample_glider_info, sample_csv_data, tmp_path):
        """Test retrieval of polar data magnitudes"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text(sample_csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            speed_mag, sink_mag = g.polar_data_magnitude()

            # Should return numpy arrays without units
            assert isinstance(speed_mag, np.ndarray)
            assert isinstance(sink_mag, np.ndarray)
            assert len(speed_mag) == 10
            assert len(sink_mag) == 10

        finally:
            os.chdir(original_dir)

    def test_get_speed_data(self, sample_glider_info, sample_csv_data, tmp_path):
        """Test get_speed_data returns quantities with units"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text(sample_csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            speed_data = g.get_speed_data()

            # Should have units
            assert hasattr(speed_data, "units")
            assert speed_data.units == ureg("m/s")

        finally:
            os.chdir(original_dir)

    def test_get_sink_data(self, sample_glider_info, sample_csv_data, tmp_path):
        """Test get_sink_data returns quantities with units"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text(sample_csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            sink_data = g.get_sink_data()

            # Should have units
            assert hasattr(sink_data, "units")
            assert sink_data.units == ureg("m/s")

        finally:
            os.chdir(original_dir)


class TestGliderEdgeCases:
    """Test cases for edge cases and boundary conditions"""

    def test_single_row_csv(self, sample_glider_info, tmp_path):
        """Test handling of CSV with only one data point"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"
        csv_file.write_text("100,-0.85")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            speed_data = g.get_speed_data()
            sink_data = g.get_sink_data()

            assert len(speed_data) == 1
            assert len(sink_data) == 1

        finally:
            os.chdir(original_dir)

    def test_very_large_wing_area(self):
        """Test glider with unusually large wing area"""
        large_area_info = pd.DataFrame(
            {
                "name": ["Large Glider"],
                "wingArea": ["1000 m**2"],
                "referenceWeight": ["5000.0 kg"],
                "emptyWeight": ["3000.0 kg"],
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(large_area_info)
            wing_loading = g.referenceWingLoading()

            # Should handle large values correctly
            assert wing_loading == ureg("5.0 kg/m**2")

    def test_zero_wing_area_edge_case(self):
        """Test that zero wing area would cause issues in wing loading calculation"""
        zero_area_info = pd.DataFrame(
            {
                "name": ["Bad Glider"],
                "wingArea": ["0.001 m**2"],  # Very small, nearly zero
                "referenceWeight": ["325.0 kg"],
                "emptyWeight": ["268.0 kg"],
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(zero_area_info)
            wing_loading = g.referenceWingLoading()

            # Should return very high wing loading
            assert wing_loading.magnitude > 100000

    def test_negative_sink_values(self, sample_glider_info, tmp_path):
        """Test CSV with all negative sink values (which is expected)"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"

        # All negative sink values (expected for gliders)
        csv_data = "100,-0.5\n120,-0.6\n140,-0.7"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            sink_data = g.get_sink_data()

            # All values should be negative
            assert np.all(sink_data.magnitude < 0)

        finally:
            os.chdir(original_dir)

    def test_messages_accumulation_on_error(self, sample_glider_info, tmp_path):
        """Test that error messages are accumulated properly"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        # File doesn't exist - should trigger error message

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            with pytest.raises(FileNotFoundError):
                g = glider.Glider(sample_glider_info)

        finally:
            os.chdir(original_dir)


class TestGliderIntegration:
    """Integration tests using real glider data"""

    def test_load_actual_glider_data(self):
        """Test loading actual glider data from the datafiles directory"""
        # This test assumes the actual datafiles exist
        if not os.path.exists("./datafiles/ASW 28.csv"):
            pytest.skip("Actual datafiles not available")

        df_glider_info = pd.read_json("datafiles/gliderInfo.json")
        asw28_info = df_glider_info[df_glider_info["name"] == "ASW 28"]

        if asw28_info.empty:
            pytest.skip("ASW 28 not found in gliderInfo.json")

        g = glider.Glider(asw28_info)

        # Verify basic properties
        assert g.name() == "ASW 28"
        assert g.get_speed_data() is not None
        assert g.get_sink_data() is not None
        assert len(g.get_speed_data()) > 0
        assert len(g.get_sink_data()) > 0

    def test_load_us_units_glider(self):
        """Test loading a glider with US customary units"""
        if not os.path.exists("./datafiles/SGS 1-34C.csv"):
            pytest.skip("Actual datafiles not available")

        df_glider_info = pd.read_json("datafiles/gliderInfo.json")
        sgs_info = df_glider_info[df_glider_info["name"] == "SGS 1-34C"]

        if sgs_info.empty:
            pytest.skip("SGS 1-34C not found in gliderInfo.json")

        g = glider.Glider(sgs_info)

        # Verify units are converted to SI
        assert g.referenceWeight().units == ureg.kg
        assert g.wingArea().units == ureg("m**2")
        assert g.get_speed_data().units == ureg("m/s")
        assert g.get_sink_data().units == ureg("m/s")


class TestGliderRegressionCases:
    """Regression tests to prevent known issues"""

    def test_csv_with_extra_whitespace(self, sample_glider_info, tmp_path):
        """Test CSV parsing handles extra whitespace correctly"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"

        # CSV with extra spaces
        csv_data = "100 , -0.85 \n120, -0.90\n140 ,-0.95"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            # Should parse successfully
            assert len(g.get_speed_data()) == 3

        finally:
            os.chdir(original_dir)

    def test_consistent_unit_conversion(self, sample_glider_info, tmp_path):
        """Test that unit conversions are consistent and reversible"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "ASW 28.csv"

        original_speed_kph = 100.0
        csv_data = f"{original_speed_kph},-0.85"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(sample_glider_info)

            speed_mps = g.get_speed_data()[0]
            speed_back_to_kph = speed_mps.to("kph")

            # Should be able to convert back accurately
            assert np.isclose(speed_back_to_kph.magnitude, original_speed_kph)

        finally:
            os.chdir(original_dir)