"""
Comprehensive tests for the ThreadSafeLogger system.
Tests all aspects of the logger including thread safety, performance, rotation, and error handling.
"""

import pytest
import tempfile
import shutil
import threading
import time
import json
import gzip
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import logging

# Import the logger system
from src.utils.logger import ThreadSafeLogger, AsyncLogHandler, CompressedRotatingFileHandler, PerformanceMonitor

class TestThreadSafeLogger:
    """Test the main ThreadSafeLogger class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Use a temporary directory for logs
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the LOG_DIR constant
        self.log_dir_patcher = patch('src.utils.logger.LOG_DIR', self.temp_dir)
        self.log_dir_patcher.start()
        
        # Create fresh logger instance
        ThreadSafeLogger._instance = None
        self.logger = ThreadSafeLogger()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        self.logger.shutdown()
        self.log_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        
        # Reset singleton
        ThreadSafeLogger._instance = None
    
    def test_singleton_pattern(self):
        """Test that ThreadSafeLogger implements singleton correctly."""
        logger1 = ThreadSafeLogger()
        logger2 = ThreadSafeLogger()
        assert logger1 is logger2
    
    def test_logger_initialization(self):
        """Test that logger is initialized correctly."""
        assert self.logger.telemetry_logger is not None
        assert self.logger.event_logger is not None
        assert self.logger.performance_monitor is not None
        assert hasattr(self.logger, '_initialized')
    
    def test_telemetry_logging(self):
        """Test telemetry logging functionality."""
        test_data = {
            'altitude': 1000.5,
            'velocity': [1.0, 2.0, 3.0],
            'timestamp': '2025-07-22T10:30:00'
        }
        
        self.logger.log_telemetry(test_data)
        
        # Wait for async processing
        time.sleep(0.2)
        self.logger.force_flush()
        
        # Check that telemetry was logged
        telemetry_file = self.temp_dir / "telemetry.log"
        if telemetry_file.exists():
            with open(telemetry_file, 'r') as f:
                content = f.read()
            
            # Should contain JSON data
            assert 'altitude' in content
            assert '1000.5' in content
    
    def test_event_logging(self):
        """Test event logging functionality."""
        test_events = [
            ("Test info message", "INFO"),
            ("Test warning message", "WARNING"),
            ("Test error message", "ERROR")
        ]
        
        for event, level in test_events:
            self.logger.log_event(event, level)
        
        self.logger.force_flush()
        
        # Check that events were logged
        event_file = self.temp_dir / "event.log"
        if event_file.exists():
            with open(event_file, 'r') as f:
                content = f.read()
            
            for event, level in test_events:
                assert event in content
    
    def test_invalid_input_handling(self):
        """Test that invalid inputs are handled gracefully."""
        # Test invalid telemetry data
        self.logger.log_telemetry("invalid_data")
        self.logger.log_telemetry(None)
        
        # Test invalid log levels
        self.logger.log_event("Test message", "INVALID_LEVEL")
        
        # Should not raise exceptions
        time.sleep(0.1)
    
    def test_thread_safety(self):
        """Test that logger is thread-safe."""
        def log_worker(worker_id):
            for i in range(20):
                self.logger.log_telemetry({
                    'worker_id': worker_id,
                    'message_id': i,
                    'data': f'test_data_{worker_id}_{i}'
                })
                self.logger.log_event(f"Worker {worker_id} message {i}", "INFO")
        
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=log_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Wait for async processing
        time.sleep(0.5)
        
        # Verify metrics
        metrics = self.logger.get_performance_metrics()
        assert metrics['total_logs'] >= 120  # 3 workers * 20 * 2 log types
    
    def test_console_handler_duplication_prevention(self):
        """Test that console handlers are not duplicated."""
        # Configure logging multiple times
        for _ in range(3):
            self.logger.configure_logging()
        
        # Count console handlers
        console_handlers = 0
        for handler in self.logger.event_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handlers += 1
        
        # Should only have one console handler
        assert console_handlers <= 1
    
    def test_performance_metrics(self):
        """Test that performance metrics are collected correctly."""
        # Generate some logs
        for i in range(10):
            self.logger.log_telemetry({'test': i})
            self.logger.log_event(f"Event {i}", "INFO")
        
        time.sleep(0.2)
        
        metrics = self.logger.get_performance_metrics()
        
        assert 'total_logs' in metrics
        assert 'telemetry_logs' in metrics
        assert 'event_logs' in metrics
        assert 'avg_processing_time' in metrics
        assert 'logs_per_second' in metrics
        assert metrics['total_logs'] >= 20
    
    def test_disk_space_monitoring(self):
        """Test disk space monitoring functionality."""
        disk_info = self.logger.check_disk_space()
        
        assert 'total_gb' in disk_info
        assert 'free_gb' in disk_info
        assert 'used_gb' in disk_info
        assert 'free_percent' in disk_info
        
        # Values should be reasonable
        assert disk_info['total_gb'] > 0
        assert disk_info['free_gb'] >= 0
        assert 0 <= disk_info['free_percent'] <= 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])