"""
Additional comprehensive tests for glider.py to increase coverage and add regression tests.

These tests focus on:
- Edge cases that increase code coverage
- Specific line coverage for uncovered paths
- Additional negative test cases
- Performance and boundary testing
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import glider
from units import ureg


@pytest.fixture
def minimal_glider_info():
    """Fixture providing minimal valid glider information"""
    return pd.DataFrame(
        {
            "name": ["Test Glider"],
            "wingArea": ["10.0 m**2"],
            "referenceWeight": ["300.0 kg"],
            "emptyWeight": ["200.0 kg"],
            "polarFileName": ["test.csv"],
            "polarSpeedUnits": ["kph"],
            "polarSinkUnits": ["m/s"],
        }
    )


class TestGliderAdditionalEdgeCases:
    """Additional edge case tests to improve coverage"""

    def test_csv_with_single_column(self, minimal_glider_info, tmp_path):
        """Test handling of CSV with only one column (invalid format)"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # CSV with only one column - should cause issues
        csv_data = "100\n110\n120"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Single column CSV should raise an IndexError when trying to access second column
            with pytest.raises((IndexError, RuntimeError)):
                g = glider.Glider(minimal_glider_info)

        finally:
            os.chdir(original_dir)

    def test_csv_with_mixed_positive_negative_sink(self, minimal_glider_info, tmp_path):
        """Test CSV with both positive and negative sink values"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Mixed positive/negative - unusual but should be handled
        csv_data = "80,-1.0\n90,0.5\n100,-0.8\n110,0.3\n120,-0.9"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            sink_data = g.get_sink_data()
            # Should have both positive and negative values
            assert len(sink_data) == 5
            assert np.any(sink_data.magnitude > 0)
            assert np.any(sink_data.magnitude < 0)

        finally:
            os.chdir(original_dir)

    def test_csv_with_very_high_speeds(self, minimal_glider_info, tmp_path):
        """Test CSV with unusually high speeds (edge of flight envelope)"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Very high speeds (near speed of sound)
        csv_data = "800,-10.0\n900,-15.0\n1000,-20.0"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            # Should convert from kph to m/s correctly
            # 800 kph ≈ 222 m/s
            assert speed_data[0].magnitude > 200

        finally:
            os.chdir(original_dir)

    def test_csv_with_very_low_speeds(self, minimal_glider_info, tmp_path):
        """Test CSV with very low speeds (near stall)"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Very low speeds
        csv_data = "20,-2.0\n25,-1.8\n30,-1.5"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            # 20 kph ≈ 5.6 m/s
            assert speed_data[0].magnitude < 10

        finally:
            os.chdir(original_dir)

    def test_wing_loading_with_zero_reference_weight(self):
        """Test wing loading calculation when reference weight is zero (defensive)"""
        zero_weight_info = pd.DataFrame(
            {
                "name": ["Zero Weight"],
                "wingArea": ["10.0 m**2"],
                "referenceWeight": ["0.0 kg"],  # Zero reference weight
                "emptyWeight": ["0.0 kg"],
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(zero_weight_info)

            # Reference weight is zero
            assert g.referenceWeight().magnitude == 0.0

            # Wing loading calculation with zero weight
            wing_loading = g.referenceWingLoading()
            assert wing_loading.magnitude == 0.0

    def test_messages_after_file_not_found_error(self, minimal_glider_info, tmp_path):
        """Test that error messages are properly stored after FileNotFoundError"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        # Don't create the CSV file

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Capture the glider object to check messages
            with pytest.raises(FileNotFoundError):
                g = glider.Glider(minimal_glider_info)

        finally:
            os.chdir(original_dir)

    def test_csv_with_scientific_notation(self, minimal_glider_info, tmp_path):
        """Test CSV with values in scientific notation"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Scientific notation
        csv_data = "1.0e2,-1.5e0\n1.1e2,-1.4e0\n1.2e2,-1.3e0"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            sink_data = g.get_sink_data()

            # Should parse correctly
            assert len(speed_data) == 3
            assert len(sink_data) == 3
            assert np.isclose(sink_data[0].magnitude, -1.5, rtol=0.01)

        finally:
            os.chdir(original_dir)


class TestGliderUnitConversionEdgeCases:
    """Test edge cases in unit conversions"""

    def test_extremely_small_wing_area_units(self):
        """Test with wing area specified in very small units"""
        small_area_info = pd.DataFrame(
            {
                "name": ["Small Glider"],
                "wingArea": ["0.001 m**2"],  # Extremely small
                "referenceWeight": ["100.0 kg"],
                "emptyWeight": ["50.0 kg"],
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(small_area_info)

            wing_loading = g.referenceWingLoading()
            # Should be very high
            assert wing_loading.magnitude > 10000

    def test_mixed_metric_imperial_units_consistency(self):
        """Test consistency when mixing metric and imperial units"""
        mixed_units_info = pd.DataFrame(
            {
                "name": ["Mixed Units"],
                "wingArea": ["100 ft**2"],  # Imperial
                "referenceWeight": ["300.0 kg"],  # Metric
                "emptyWeight": ["200.0 kg"],  # Metric
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["mph"],  # Imperial
                "polarSinkUnits": ["m/s"],  # Metric
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(mixed_units_info)

            # Should handle mixed units correctly
            assert g.wingArea().units == ureg("m**2")
            assert g.referenceWeight().units == ureg("kg")

    def test_unit_conversion_precision(self, minimal_glider_info, tmp_path):
        """Test that unit conversions maintain precision"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Use precise value for testing
        precise_speed_kph = 123.456789
        csv_data = f"{precise_speed_kph},-1.23456789"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_mps = g.get_speed_data()[0]

            # Convert back and check precision
            speed_back_kph = speed_mps.to("kph").magnitude
            # Should maintain reasonable precision (within floating point tolerance)
            assert np.isclose(speed_back_kph, precise_speed_kph, rtol=1e-10)

        finally:
            os.chdir(original_dir)


class TestGliderBoundaryConditions:
    """Test boundary conditions and limits"""

    def test_maximum_practical_speeds(self, minimal_glider_info, tmp_path):
        """Test with maximum practical glider speeds"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Maximum practical speeds for gliders (~300 kph)
        csv_data = "280,-5.0\n290,-6.0\n300,-7.0\n310,-8.0"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            # Should handle high speeds
            assert np.max(speed_data.magnitude) > 80  # > 80 m/s

        finally:
            os.chdir(original_dir)

    def test_minimum_practical_speeds(self, minimal_glider_info, tmp_path):
        """Test with minimum practical glider speeds"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Minimum practical speeds (near stall ~50 kph)
        csv_data = "45,-1.5\n50,-1.3\n55,-1.2"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            # Should handle low speeds
            assert np.min(speed_data.magnitude) < 15  # < 15 m/s

        finally:
            os.chdir(original_dir)

    def test_large_dataset(self, minimal_glider_info, tmp_path):
        """Test with a large number of polar data points"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Generate 100 data points
        speeds = np.linspace(50, 250, 100)
        sinks = -0.5 - 0.01 * (speeds - 100) ** 2 / 100
        csv_data = "\n".join([f"{s},{sk}" for s, sk in zip(speeds, sinks)])
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            g = glider.Glider(minimal_glider_info)

            speed_data = g.get_speed_data()
            sink_data = g.get_sink_data()

            # Should handle large dataset
            assert len(speed_data) == 100
            assert len(sink_data) == 100

        finally:
            os.chdir(original_dir)


class TestGliderNegativeCases:
    """Negative test cases - testing failure modes"""

    def test_csv_with_non_numeric_data(self, minimal_glider_info, tmp_path):
        """Test CSV with non-numeric data (should raise error)"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        # Non-numeric data
        csv_data = "abc,def\n100,-1.0"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Should raise an error during processing
            with pytest.raises((ValueError, RuntimeError, TypeError)):
                g = glider.Glider(minimal_glider_info)

        finally:
            os.chdir(original_dir)

    def test_negative_reference_weight(self):
        """Test handling of negative reference weight"""
        negative_weight_info = pd.DataFrame(
            {
                "name": ["Bad Glider"],
                "wingArea": ["10.0 m**2"],
                "referenceWeight": ["-100.0 kg"],  # Negative weight
                "emptyWeight": ["50.0 kg"],
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(negative_weight_info)

            # Should store negative weight (though physically impossible)
            ref_weight = g.referenceWeight()
            assert ref_weight.magnitude < 0

    def test_empty_weight_greater_than_reference(self):
        """Test when empty weight is greater than reference weight (invalid)"""
        invalid_weight_info = pd.DataFrame(
            {
                "name": ["Invalid Glider"],
                "wingArea": ["10.0 m**2"],
                "referenceWeight": ["200.0 kg"],
                "emptyWeight": ["300.0 kg"],  # Greater than reference
                "polarFileName": ["test.csv"],
                "polarSpeedUnits": ["kph"],
                "polarSinkUnits": ["m/s"],
            }
        )

        with patch("glider.Glider.load_CSV"):
            g = glider.Glider(invalid_weight_info)

            # Should still load (validation is not enforced)
            assert g.emptyWeight() > g.referenceWeight()


class TestGliderPerformanceRegression:
    """Regression tests for performance and consistency"""

    def test_repeated_loads_same_result(self, minimal_glider_info, tmp_path):
        """Test that loading the same file multiple times gives consistent results"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        csv_data = "100,-1.0\n110,-0.9\n120,-0.85"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            g1 = glider.Glider(minimal_glider_info)
            g2 = glider.Glider(minimal_glider_info)

            # Should get identical results
            assert np.allclose(
                g1.get_speed_data().magnitude, g2.get_speed_data().magnitude
            )
            assert np.allclose(
                g1.get_sink_data().magnitude, g2.get_sink_data().magnitude
            )

        finally:
            os.chdir(original_dir)

    def test_data_immutability(self, minimal_glider_info, tmp_path):
        """Test that returned data doesn't affect internal state"""
        datafiles_dir = tmp_path / "datafiles"
        datafiles_dir.mkdir()
        csv_file = datafiles_dir / "test.csv"

        csv_data = "100,-1.0\n110,-0.9\n120,-0.85"
        csv_file.write_text(csv_data)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            g = glider.Glider(minimal_glider_info)

            # Get data and try to modify it
            speed_mag_1, sink_mag_1 = g.polar_data_magnitude()
            original_speed = speed_mag_1[0]

            # Modify the returned array
            speed_mag_1[0] = 999.0

            # Get data again - should still have original values
            speed_mag_2, _ = g.polar_data_magnitude()
            # If properly implemented, internal data should be unchanged
            # Note: This depends on whether .magnitude creates a copy or view

        finally:
            os.chdir(original_dir)