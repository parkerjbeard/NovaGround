"""
Core telemetry system tests without GUI dependencies.
Tests telemetry data validation and core functionality.
"""

import pytest
import time
import threading
import numpy as np

# Import the telemetry system components
from src.utils.telemetry_data import TelemetryData, create_telemetry_from_dict, merge_telemetry_data
from src.gui.mission_control_panel import CircularBuffer, TelemetryValidator

class TestTelemetryData:
    """Test the enhanced TelemetryData class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.telemetry = TelemetryData()
    
    def test_default_initialization(self):
        """Test that TelemetryData initializes with safe defaults."""
        assert self.telemetry.position == (0.0, 0.0, 0.0)
        assert self.telemetry.orientation == (0.0, 0.0, 0.0)
        assert self.telemetry.velocity == (0.0, 0.0, 0.0)
        assert self.telemetry.acceleration == (0.0, 0.0, 9.81)
        assert self.telemetry.voltage == 12000
        assert self.telemetry.data_quality == 1.0
        assert len(self.telemetry.validation_errors) == 0
        assert self.telemetry.status_flags['system_health'] is True
    
    def test_valid_data_update(self):
        """Test updating with valid telemetry data."""
        valid_data = {
            'position': [10.0, 20.0, 1000.0],
            'velocity': [5.0, 10.0, 15.0],
            'voltage': 11500,
            'status_flags': {'motor_failure': False, 'system_health': True}
        }
        
        self.telemetry.update(valid_data)
        
        assert self.telemetry.position == (10.0, 20.0, 1000.0)
        assert self.telemetry.velocity == (5.0, 10.0, 15.0)
        assert self.telemetry.voltage == 11500
        assert self.telemetry.altitude == 1000.0
        assert len(self.telemetry.validation_errors) == 0
        assert self.telemetry.data_quality > 0.9
    
    def test_invalid_data_validation(self):
        """Test validation of invalid telemetry data."""
        invalid_data = {
            'position': [1e7, 1e7, 200000],  # Out of range coordinates
            'velocity': [3000, 0, 0],        # Velocity too high
            'voltage': 25000,                # Voltage too high
            'altitude': 150000               # Altitude too high
        }
        
        self.telemetry.update(invalid_data)
        
        # Should have validation errors
        assert len(self.telemetry.validation_errors) > 0
        assert self.telemetry.data_quality < 1.0
    
    def test_thread_safety(self):
        """Test that TelemetryData is thread-safe."""
        results = []
        
        def update_worker(worker_id):
            for i in range(50):  # Reduced for stability
                data = {
                    'position': [worker_id, i, i * 10],
                    'voltage': 12000 + worker_id * 100 + i
                }
                try:
                    self.telemetry.update(data)
                    results.append(self.telemetry.get_summary())
                except Exception as e:
                    results.append({'error': str(e)})
        
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(3):  # Reduced thread count
            thread = threading.Thread(target=update_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have processed updates without corruption
        assert len(results) >= 100
        assert self.telemetry.data_quality >= 0.0
    
    def test_computed_fields(self):
        """Test computation of derived fields."""
        data = {
            'position': [10, 20, 500],
            'velocity': [3, 4, 5],
            'acceleration': [1, 1, 10]
        }
        self.telemetry.update(data)
        
        # Check computed fields
        assert self.telemetry.altitude == 500.0
        
        expected_vel_mag = np.sqrt(3**2 + 4**2 + 5**2)
        assert abs(self.telemetry.velocity_magnitude - expected_vel_mag) < 0.01
        
        expected_accel_mag = np.sqrt(1**2 + 1**2 + 10**2)
        assert abs(self.telemetry.acceleration_magnitude - expected_accel_mag) < 0.01
    
    def test_data_validation_and_quality(self):
        """Test data quality scoring."""
        # Start with good data
        good_data = {
            'position': [0, 0, 100],
            'velocity': [0, 0, 10],
            'voltage': 12000,
            'status_flags': {'motor_failure': False, 'system_health': True}
        }
        self.telemetry.update(good_data)
        assert self.telemetry.data_quality > 0.9
        
        # Add motor failure
        bad_data = {
            'status_flags': {'motor_failure': True, 'system_health': False}
        }
        self.telemetry.update(bad_data)
        assert self.telemetry.data_quality < 0.8  # Should be reduced

class TestCircularBuffer:
    """Test the CircularBuffer class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.buffer = CircularBuffer(maxsize=5)
    
    def test_buffer_operations(self):
        """Test basic buffer operations."""
        # Add items
        for i in range(3):
            self.buffer.append(i)
        
        data = self.buffer.get_data()
        assert data == [0, 1, 2]
        
        # Test overflow
        for i in range(3, 8):
            self.buffer.append(i)
        
        data = self.buffer.get_data()
        assert len(data) == 5
        assert data == [3, 4, 5, 6, 7]  # Oldest items dropped
    
    def test_thread_safety(self):
        """Test that CircularBuffer is thread-safe."""
        def worker(start, count):
            for i in range(start, start + count):
                self.buffer.append(i)
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i * 100, 20))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Buffer should have 5 items (its max size)
        data = self.buffer.get_data()
        assert len(data) == 5

class TestTelemetryValidator:
    """Test the TelemetryValidator class."""
    
    def test_valid_data_validation(self):
        """Test validation of valid data."""
        valid_data = {
            'altitude': 1000.0,
            'velocity': [10, 20, 30],
            'acceleration': [1, 2, 15],
            'voltage': 12000
        }
        
        result = TelemetryValidator.validate_telemetry(valid_data)
        
        assert 'validation_warnings' in result
        assert len(result['validation_warnings']) == 0
        assert result['altitude'] == 1000.0
    
    def test_out_of_range_validation(self):
        """Test validation of out-of-range data."""
        invalid_data = {
            'altitude': 200000,     # Too high
            'velocity': [5000, 0, 0],  # Too fast
            'acceleration': [500, 0, 0],  # Too high
            'voltage': 30000        # Too high
        }
        
        result = TelemetryValidator.validate_telemetry(invalid_data)
        
        assert 'validation_warnings' in result
        assert len(result['validation_warnings']) > 0

class TestTelemetryUtilities:
    """Test utility functions for telemetry data management."""
    
    def test_create_telemetry_from_dict(self):
        """Test creating TelemetryData from dictionary."""
        data = {
            'position': [10, 20, 500],
            'velocity': [1, 2, 3],
            'voltage': 11000
        }
        
        telemetry = create_telemetry_from_dict(data)
        
        assert isinstance(telemetry, TelemetryData)
        assert telemetry.position == (10, 20, 500)
        assert telemetry.voltage == 11000
    
    def test_create_telemetry_with_invalid_data(self):
        """Test creating TelemetryData with invalid data."""
        invalid_data = {
            'position': "invalid",
            'voltage': "not a number"
        }
        
        telemetry = create_telemetry_from_dict(invalid_data)
        
        # Should return a TelemetryData object with errors
        assert isinstance(telemetry, TelemetryData)
        assert len(telemetry.validation_errors) > 0
    
    def test_merge_telemetry_data(self):
        """Test merging two TelemetryData instances."""
        # Create primary data (valid)
        primary_data = {
            'position': [10, 20, 500],
            'voltage': 12000,
            'status_flags': {'system_health': True}
        }
        primary = create_telemetry_from_dict(primary_data)
        
        # Create secondary data (fallback)
        secondary_data = {
            'position': [0, 0, 0],
            'voltage': 11000,
            'velocity': [5, 10, 15]
        }
        secondary = create_telemetry_from_dict(secondary_data)
        
        # Merge
        merged = merge_telemetry_data(primary, secondary)
        
        # Should prefer primary data
        assert merged.position == (10, 20, 500)
        assert merged.voltage == 12000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])