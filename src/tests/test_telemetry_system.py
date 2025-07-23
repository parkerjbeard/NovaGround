"""
Comprehensive tests for the enhanced telemetry system.
Tests telemetry data validation, mission control panel integration, and thread safety.
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest

# Import the telemetry system components
from src.utils.telemetry_data import TelemetryData, create_telemetry_from_dict, merge_telemetry_data
from src.gui.mission_control_panel import (
    MissionControlPanel, CircularBuffer, TelemetryValidator
)

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
        
        # Values should be clamped or rejected
        assert self.telemetry.voltage != 25000  # Should be rejected
    
    def test_thread_safety(self):
        """Test that TelemetryData is thread-safe."""
        results = []
        
        def update_worker(worker_id):
            for i in range(100):
                data = {
                    'position': [worker_id, i, i * 10],
                    'voltage': 12000 + worker_id * 100 + i
                }
                self.telemetry.update(data)
                results.append(self.telemetry.get_summary())
        
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=update_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have processed all updates without corruption
        assert len(results) == 500
        assert self.telemetry.data_quality >= 0.0
    
    def test_data_quality_calculation(self):
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
    
    def test_data_freshness(self):
        """Test data freshness checking."""
        assert self.telemetry.is_data_fresh(max_age_seconds=5.0)
        
        # Simulate old data
        self.telemetry.last_update_time = time.time() - 10.0
        assert not self.telemetry.is_data_fresh(max_age_seconds=5.0)
    
    def test_safe_copy(self):
        """Test thread-safe copying."""
        data = {
            'position': [1, 2, 3],
            'velocity': [4, 5, 6],
            'voltage': 11000
        }
        self.telemetry.update(data)
        
        copy = self.telemetry.get_safe_copy()
        
        assert copy.position == self.telemetry.position
        assert copy.velocity == self.telemetry.velocity
        assert copy.voltage == self.telemetry.voltage
        assert copy is not self.telemetry  # Different object
    
    def test_reset_to_defaults(self):
        """Test resetting telemetry to safe defaults."""
        # Set some data
        data = {
            'position': [100, 200, 1000],
            'velocity': [10, 20, 30],
            'voltage': 11000
        }
        self.telemetry.update(data)
        
        # Reset
        self.telemetry.reset_to_defaults()
        
        # Should be back to defaults
        assert self.telemetry.position == (0.0, 0.0, 0.0)
        assert self.telemetry.velocity == (0.0, 0.0, 0.0)
        assert self.telemetry.voltage == 12000
        assert len(self.telemetry.validation_errors) == 0

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
            thread = threading.Thread(target=worker, args=(i * 100, 50))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Buffer should have 5 items (its max size)
        data = self.buffer.get_data()
        assert len(data) == 5
    
    def test_clear(self):
        """Test buffer clearing."""
        for i in range(3):
            self.buffer.append(i)
        
        assert len(self.buffer.get_data()) == 3
        
        self.buffer.clear()
        assert len(self.buffer.get_data()) == 0

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
        
        # Check that warnings contain expected messages
        warnings_text = ' '.join(result['validation_warnings'])
        assert 'Altitude out of range' in warnings_text
        assert 'Velocity' in warnings_text or 'out of range' in warnings_text

@pytest.fixture
def qapp():
    """Create QApplication for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

class TestMissionControlPanel:
    """Test the enhanced MissionControlPanel class."""
    
    def setup_method(self, qapp):
        """Setup for each test method."""
        self.panel = MissionControlPanel()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        if hasattr(self, 'panel'):
            self.panel.close()
    
    def test_panel_initialization(self):
        """Test that MissionControlPanel initializes correctly."""
        assert self.panel.max_data_points == 1000
        assert hasattr(self.panel, 'telemetry_buffer')
        assert hasattr(self, 'graph_data')
        assert hasattr(self.panel, 'performance_monitor')
    
    def test_telemetry_update(self):
        """Test telemetry data update functionality."""
        test_data = {
            'altitude': 500.0,
            'velocity': [5, 10, 15],
            'acceleration': [1, 2, 20],
            'voltage': 11500,
            'timestamp': time.time(),
            'status_flags': {'motor_failure': False, 'system_health': True}
        }
        
        # Update telemetry
        self.panel.update_telemetry(test_data)
        
        # Check that data was stored
        buffer_data = self.panel.telemetry_buffer.get_data()
        assert len(buffer_data) == 1
        
        # Check that current telemetry was updated
        assert 'altitude' in self.panel.current_telemetry
        assert self.panel.current_telemetry['altitude'] == 500.0
    
    def test_invalid_telemetry_handling(self):
        """Test handling of invalid telemetry data."""
        invalid_data = "not a dictionary"
        
        # Should not raise an exception
        self.panel.update_telemetry(invalid_data)
        
        # Should handle gracefully
        assert len(self.panel.telemetry_buffer.get_data()) == 0
    
    def test_performance_monitoring(self):
        """Test telemetry performance monitoring."""
        # Send multiple telemetry updates
        for i in range(10):
            data = {
                'altitude': i * 100,
                'velocity': [i, i*2, i*3],
                'voltage': 12000 + i
            }
            self.panel.update_telemetry(data)
            time.sleep(0.01)  # Small delay
        
        # Check performance metrics
        stats = self.panel.get_telemetry_statistics()
        
        assert 'update_count' in stats
        assert 'telemetry_rate_hz' in stats
        assert stats['update_count'] >= 10
        assert stats['telemetry_rate_hz'] > 0
    
    def test_graph_data_management(self):
        """Test that graph data is managed properly."""
        # Send data to fill graphs
        for i in range(20):
            data = {
                'altitude': i * 50,
                'velocity': [0, 0, i],
                'acceleration': [0, 0, 9.81 + i],
                'timestamp': time.time() + i
            }
            self.panel.update_telemetry(data)
        
        # Check graph data buffers
        altitude_data = self.panel.graph_data['altitude'].get_data()
        velocity_data = self.panel.graph_data['velocity'].get_data()
        acceleration_data = self.panel.graph_data['acceleration'].get_data()
        
        assert len(altitude_data) == 20
        assert len(velocity_data) == 20
        assert len(acceleration_data) == 20
        
        # Check data format (timestamp, value)
        assert len(altitude_data[0]) == 2
        assert isinstance(altitude_data[0][0], float)  # timestamp
        assert isinstance(altitude_data[0][1], (int, float))  # value
    
    def test_clear_graphs(self):
        """Test graph clearing functionality."""
        # Add some data
        for i in range(5):
            data = {
                'altitude': i * 100,
                'velocity': [0, 0, i],
                'acceleration': [0, 0, 9.81]
            }
            self.panel.update_telemetry(data)
        
        # Verify data exists
        assert len(self.panel.graph_data['altitude'].get_data()) > 0
        
        # Clear graphs
        self.panel.clear_graphs()
        
        # Verify data is cleared
        assert len(self.panel.graph_data['altitude'].get_data()) == 0
        assert len(self.panel.graph_data['velocity'].get_data()) == 0
        assert len(self.panel.graph_data['acceleration'].get_data()) == 0

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

class TestTelemetryIntegration:
    """Integration tests for the complete telemetry system."""
    
    def test_end_to_end_telemetry_flow(self, qapp):
        """Test complete telemetry data flow from creation to display."""
        # Create panel
        panel = MissionControlPanel()
        
        try:
            # Create telemetry data
            raw_data = {
                'position': [45.0, -122.0, 1500.0],
                'orientation': [10.0, 20.0, 5.0],
                'velocity': [50.0, 30.0, 100.0],
                'acceleration': [2.0, 1.0, 25.0],
                'voltage': 12500,
                'status_flags': {
                    'motor_failure': False,
                    'sensor_error': False,
                    'system_health': True
                }
            }
            
            # Create TelemetryData object
            telemetry = create_telemetry_from_dict(raw_data)
            
            # Verify data quality
            assert telemetry.is_data_valid()
            assert telemetry.is_data_fresh()
            
            # Update panel with telemetry
            telemetry_dict = telemetry.to_dict()
            panel.update_telemetry(telemetry_dict)
            
            # Wait for processing
            time.sleep(0.1)
            
            # Verify panel updated
            stats = panel.get_telemetry_statistics()
            assert stats['update_count'] >= 1
            assert 'current_telemetry' in stats
            assert stats['current_telemetry']['altitude'] == 1500.0
            
        finally:
            panel.close()
    
    def test_high_frequency_telemetry(self, qapp):
        """Test handling of high-frequency telemetry updates."""
        panel = MissionControlPanel()
        
        try:
            start_time = time.time()
            
            # Send 100 rapid updates
            for i in range(100):
                data = {
                    'altitude': i * 10,
                    'velocity': [i, i*2, i*3],
                    'acceleration': [1, 2, 9.81 + i*0.1],
                    'voltage': 12000 + i,
                    'timestamp': time.time()
                }
                panel.update_telemetry(data)
            
            # Wait for processing
            time.sleep(0.5)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should handle high frequency efficiently
            assert duration < 2.0  # Should complete quickly
            
            # Check statistics
            stats = panel.get_telemetry_statistics()
            assert stats['update_count'] >= 100
            assert stats['telemetry_rate_hz'] > 10  # Should show high rate
            
        finally:
            panel.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])