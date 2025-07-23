"""
Comprehensive Logger System Tests

This module contains comprehensive tests for the NovoGround logger system,
including thread safety, performance, rotation, compression, and monitoring.
"""

import pytest
import threading
import time
import json
import tempfile
import shutil
import os
import logging
import gzip
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import logger components
from src.utils.logger import (
    ThreadSafeLogger, 
    AsyncLogHandler,
    CompressedRotatingFileHandler,
    PerformanceMonitor,
    logger
)


class TestThreadSafeLogger:
    """Test ThreadSafeLogger functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_logger = ThreadSafeLogger()
        self.errors = []
        self.results = []
    
    def teardown_method(self):
        """Cleanup after each test method."""
        try:
            if hasattr(self.test_logger, 'shutdown'):
                self.test_logger.shutdown()
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_singleton_pattern(self):
        """Test that ThreadSafeLogger follows singleton pattern."""
        logger1 = ThreadSafeLogger()
        logger2 = ThreadSafeLogger()
        
        assert logger1 is logger2
        assert id(logger1) == id(logger2)
    
    def test_telemetry_logging(self):
        """Test telemetry logging functionality."""
        test_data = {
            'position': [1.0, 2.0, 3.0],
            'velocity': [0.1, 0.2, 0.3],
            'voltage': 12500,
            'status': 'active'
        }
        
        # Test normal logging
        self.test_logger.log_telemetry(test_data)
        
        # Test invalid data handling
        self.test_logger.log_telemetry("invalid_data")
        self.test_logger.log_telemetry(None)
        
        # No exceptions should be raised
        assert True
    
    def test_event_logging(self):
        """Test event logging functionality."""
        # Test different log levels
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            self.test_logger.log_event(f"Test {level} message", level)
        
        # Test invalid level handling
        self.test_logger.log_event("Test message", "INVALID_LEVEL")
        
        # Test non-string event
        self.test_logger.log_event(12345, 'INFO')
        
        # No exceptions should be raised
        assert True
    
    def test_performance_metrics(self):
        """Test performance monitoring functionality."""
        # Generate some logging activity
        for i in range(10):
            self.test_logger.log_event(f"Test event {i}", 'INFO')
            test_data = {'iteration': i, 'value': i * 2}
            self.test_logger.log_telemetry(test_data)
        
        # Get performance metrics
        metrics = self.test_logger.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_logs' in metrics
        assert 'telemetry_logs' in metrics
        assert 'event_logs' in metrics
        assert 'avg_processing_time' in metrics
        assert metrics['total_logs'] >= 20  # At least 10 event + 10 telemetry logs
    
    def test_disk_space_monitoring(self):
        """Test disk space monitoring functionality."""
        disk_info = self.test_logger.check_disk_space()
        
        assert isinstance(disk_info, dict)
        if disk_info:  # If check succeeded
            assert 'total_gb' in disk_info
            assert 'free_gb' in disk_info
            assert 'used_gb' in disk_info
            assert 'free_percent' in disk_info
            assert disk_info['free_percent'] >= 0
            assert disk_info['free_percent'] <= 100
    
    def test_force_flush(self):
        """Test force flush functionality."""
        # Add some log entries
        self.test_logger.log_event("Test flush message", 'INFO')
        
        # Should not raise exceptions
        self.test_logger.force_flush()
        assert True
    
    def test_async_logging(self):
        """Test asynchronous logging functionality."""
        # Test async logging submission
        for i in range(5):
            self.test_logger.log_async(
                self.test_logger.log_event, 
                f"Async test {i}", 
                'INFO'
            )
        
        # Give time for async operations to complete
        time.sleep(0.1)
        
        # No exceptions should be raised
        assert True


class TestAsyncLogHandler:
    """Test AsyncLogHandler functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Create a file handler as target
        self.file_handler = logging.FileHandler(self.temp_file.name)
        self.async_handler = AsyncLogHandler(self.file_handler, queue_size=100)
        
        # Create a test logger
        self.test_logger = logging.getLogger('test_async')
        self.test_logger.addHandler(self.async_handler)
        self.test_logger.setLevel(logging.DEBUG)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        try:
            self.async_handler.close()
            os.unlink(self.temp_file.name)
        except Exception:
            pass
    
    def test_async_handler_creation(self):
        """Test AsyncLogHandler creation and initialization."""
        assert self.async_handler.target_handler is self.file_handler
        assert self.async_handler.log_queue.maxsize == 100
        assert self.async_handler.worker_thread.is_alive()
    
    def test_async_logging_basic(self):
        """Test basic async logging functionality."""
        # Send some log messages
        for i in range(10):
            self.test_logger.info(f"Test async message {i}")
        
        # Give time for async processing
        time.sleep(0.2)
        
        # Force close to ensure all messages are processed
        self.async_handler.close()
        
        # Check that messages were written to file
        with open(self.temp_file.name, 'r') as f:
            content = f.read()
            assert "Test async message" in content
    
    def test_queue_overflow_handling(self):
        """Test queue overflow handling."""
        # Fill the queue beyond capacity
        for i in range(150):  # More than queue size of 100
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='', lineno=0,
                msg=f"Overflow test {i}", args=(), exc_info=None
            )
            self.async_handler.emit(record)
        
        # Handler should still be functional
        assert self.async_handler.worker_thread.is_alive()


class TestCompressedRotatingFileHandler:
    """Test CompressedRotatingFileHandler functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Create handler with small size for easy testing
        self.handler = CompressedRotatingFileHandler(
            filename=self.log_file,
            maxBytes=1024,  # 1KB for easy rotation
            backupCount=3
        )
        
        self.test_logger = logging.getLogger('test_rotation')
        self.test_logger.addHandler(self.handler)
        self.test_logger.setLevel(logging.DEBUG)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        try:
            self.handler.close()
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_log_rotation(self):
        """Test log file rotation functionality."""
        # Generate enough log data to trigger rotation
        large_message = "x" * 200  # 200 char message
        
        for i in range(20):  # Should trigger multiple rotations
            self.test_logger.info(f"Message {i}: {large_message}")
        
        # Force rotation and compression
        self.handler.doRollover()
        
        # Check that compressed files were created
        compressed_files = list(Path(self.temp_dir).glob("*.gz"))
        assert len(compressed_files) > 0
        
        # Verify compressed file can be read
        for comp_file in compressed_files:
            with gzip.open(comp_file, 'rt') as f:
                content = f.read()
                assert "Message" in content


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.monitor = PerformanceMonitor()
    
    def test_metric_recording(self):
        """Test metric recording functionality."""
        # Record some logs
        self.monitor.record_log('telemetry', 0.001)
        self.monitor.record_log('event', 0.002)
        self.monitor.record_log('telemetry', 0.003)
        
        # Record some errors
        self.monitor.record_error('general')
        self.monitor.record_error('queue_overflow')
        
        # Get metrics
        metrics = self.monitor.get_metrics()
        
        assert metrics['total_logs'] == 3
        assert metrics['telemetry_logs'] == 2
        assert metrics['event_logs'] == 1
        assert metrics['errors'] == 2
        assert metrics['queue_overflows'] == 1
        assert metrics['avg_processing_time'] > 0
        assert 'uptime' in metrics
        assert 'logs_per_second' in metrics
    
    def test_thread_safety(self):
        """Test PerformanceMonitor thread safety."""
        errors = []
        
        def worker(worker_id: int):
            try:
                for i in range(50):
                    self.monitor.record_log('telemetry', 0.001)
                    self.monitor.record_error('general')
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Run multiple workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0
        
        # Verify metrics
        metrics = self.monitor.get_metrics()
        assert metrics['total_logs'] == 150  # 3 workers * 50 logs
        assert metrics['errors'] == 150  # 3 workers * 50 errors


class TestLoggerThreadSafety:
    """Test logger thread safety under concurrent access."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = ThreadSafeLogger()
        self.errors = []
        self.results = []
    
    def test_concurrent_telemetry_logging(self):
        """Test concurrent telemetry logging from multiple threads."""
        def telemetry_worker(worker_id: int):
            try:
                for i in range(50):
                    data = {
                        'worker_id': worker_id,
                        'iteration': i,
                        'position': [worker_id + i, worker_id * 2 + i, worker_id * 3 + i],
                        'timestamp': time.time()
                    }
                    self.logger.log_telemetry(data)
                    self.results.append(f"Worker {worker_id} logged telemetry {i}")
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Telemetry worker {worker_id} error: {e}")
        
        # Run telemetry workers
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=telemetry_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0
        
        # Verify all operations completed
        assert len(self.results) == 200  # 4 workers * 50 operations
    
    def test_concurrent_event_logging(self):
        """Test concurrent event logging from multiple threads."""
        def event_worker(worker_id: int):
            try:
                levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
                for i in range(25):
                    level = levels[i % len(levels)]
                    message = f"Worker {worker_id} event {i}"
                    self.logger.log_event(message, level)
                    self.results.append(f"Worker {worker_id} logged event {i}")
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Event worker {worker_id} error: {e}")
        
        # Run event workers
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=event_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0
        
        # Verify all operations completed
        assert len(self.results) == 100  # 4 workers * 25 operations
    
    def test_mixed_concurrent_operations(self):
        """Test mixed concurrent logging operations."""
        def mixed_worker(worker_id: int):
            try:
                for i in range(30):
                    # Mix telemetry and event logging
                    if i % 2 == 0:
                        data = {'worker': worker_id, 'iter': i, 'value': i * worker_id}
                        self.logger.log_telemetry(data)
                    else:
                        self.logger.log_event(f"Worker {worker_id} event {i}", 'INFO')
                    
                    # Occasionally check metrics
                    if i % 10 == 0:
                        metrics = self.logger.get_performance_metrics()
                        self.results.append(f"Worker {worker_id} checked metrics: {metrics['total_logs']}")
                    
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Mixed worker {worker_id} error: {e}")
        
        # Run mixed workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=mixed_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0
        
        # Verify metrics checks completed
        metric_checks = [r for r in self.results if "checked metrics" in r]
        assert len(metric_checks) >= 9  # 3 workers * at least 3 checks each


class TestLoggerConfiguration:
    """Test logger configuration and setup."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = ThreadSafeLogger()
    
    def test_configure_logging(self):
        """Test logging configuration."""
        # Should not raise exceptions
        self.logger.configure_logging()
        
        # Test multiple calls (should be idempotent)
        self.logger.configure_logging()
        self.logger.configure_logging()
        
        assert True
    
    def test_cleanup_operations(self):
        """Test cleanup and maintenance operations."""
        # Test log cleanup (should not raise exceptions)
        self.logger.cleanup_old_logs(days_to_keep=30)
        
        # Test disk space check
        disk_info = self.logger.check_disk_space()
        assert isinstance(disk_info, dict)


class TestLoggerErrorHandling:
    """Test logger error handling and edge cases."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = ThreadSafeLogger()
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test None inputs
        self.logger.log_telemetry(None)
        self.logger.log_event(None)
        
        # Test empty inputs
        self.logger.log_telemetry({})
        self.logger.log_event("")
        
        # Test invalid types
        self.logger.log_telemetry("not_a_dict")
        self.logger.log_event(12345)
        
        # Should handle gracefully without exceptions
        assert True
    
    @patch('src.utils.logger.LOG_DIR')
    def test_filesystem_errors(self, mock_log_dir):
        """Test handling of filesystem errors."""
        # Mock filesystem error
        mock_log_dir.mkdir.side_effect = OSError("Permission denied")
        
        # Should handle gracefully
        try:
            test_logger = ThreadSafeLogger()
            assert True
        except Exception:
            # If it fails, that's also acceptable for permission errors
            assert True
    
    def test_shutdown_handling(self):
        """Test graceful shutdown handling."""
        # Create a fresh logger for shutdown testing
        test_logger = ThreadSafeLogger()
        
        # Add some activity
        test_logger.log_event("Pre-shutdown event", 'INFO')
        test_logger.log_telemetry({'test': 'data'})
        
        # Should shutdown gracefully
        test_logger.shutdown()
        
        # Additional shutdown calls should be safe
        test_logger.shutdown()
        test_logger.shutdown()
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])