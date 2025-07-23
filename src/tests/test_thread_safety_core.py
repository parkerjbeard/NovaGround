"""
Core Thread Safety Tests (without GUI dependencies)

This module contains tests to verify thread safety of core components
that don't require PyQt5 or GUI dependencies.
"""

import pytest
import threading
import time
import random
from typing import List, Dict, Any

# Import only core thread-safe components
from src.utils.telemetry_data import TelemetryData
from src.gui.mission_control_panel import CircularBuffer


class TestTelemetryDataThreadSafety:
    """Test thread safety of TelemetryData class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.telemetry = TelemetryData()
        self.results = []
        self.errors = []
    
    def test_concurrent_updates(self):
        """Test concurrent updates to TelemetryData."""
        def update_worker(worker_id: int, iterations: int):
            try:
                for i in range(iterations):
                    data = {
                        'position': [worker_id * 100 + i, worker_id * 200 + i, worker_id * 300 + i],
                        'velocity': [worker_id + i, worker_id * 2 + i, worker_id * 3 + i],
                        'voltage': 12000 + worker_id * 100 + i,
                        'status_flags': {'worker_id': worker_id, 'iteration': i}
                    }
                    self.telemetry.update(data)
                    self.results.append(f"Worker {worker_id} completed iteration {i}")
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                self.errors.append(f"Worker {worker_id} error: {e}")
        
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=update_worker, args=(worker_id, 20))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify we got expected number of updates
        assert len(self.results) == 100  # 5 workers * 20 iterations
        
        # Verify telemetry data is in a valid state
        assert self.telemetry.data_quality >= 0.0
        assert isinstance(self.telemetry.position, tuple)
        assert len(self.telemetry.position) == 3
    
    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes to TelemetryData."""
        read_results = []
        write_count = 0
        
        def reader_worker(worker_id: int):
            try:
                for i in range(50):
                    # Read operations
                    copy = self.telemetry.get_safe_copy()
                    summary = self.telemetry.get_summary()
                    is_fresh = self.telemetry.is_data_fresh()
                    is_valid = self.telemetry.is_data_valid()
                    
                    read_results.append({
                        'worker_id': worker_id,
                        'iteration': i,
                        'position': copy.position if copy else None,
                        'summary_altitude': summary.get('altitude', 0),
                        'is_fresh': is_fresh,
                        'is_valid': is_valid
                    })
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Reader {worker_id} error: {e}")
        
        def writer_worker(worker_id: int):
            nonlocal write_count
            try:
                for i in range(30):
                    data = {
                        'position': [random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(0, 1000)],
                        'velocity': [random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50)],
                        'acceleration': [random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)],
                        'voltage': random.randint(10000, 15000)
                    }
                    self.telemetry.update(data)
                    write_count += 1
                    time.sleep(0.002)
            except Exception as e:
                self.errors.append(f"Writer {worker_id} error: {e}")
        
        # Start reader and writer threads
        threads = []
        
        # 3 reader threads
        for i in range(3):
            thread = threading.Thread(target=reader_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 2 writer threads
        for i in range(2):
            thread = threading.Thread(target=writer_worker, args=(i + 10,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify read operations completed
        assert len(read_results) == 150  # 3 readers * 50 iterations
        
        # Verify write operations completed
        assert write_count == 60  # 2 writers * 30 iterations
        
        # Verify all reads returned valid data
        for result in read_results:
            assert result['position'] is not None
            assert isinstance(result['summary_altitude'], (int, float))
            assert isinstance(result['is_fresh'], bool)
            assert isinstance(result['is_valid'], bool)
    
    def test_thread_safe_validation(self):
        """Test that validation works correctly under concurrent access."""
        validation_results = []
        
        def validation_worker(worker_id: int):
            try:
                for i in range(30):
                    # Mix of valid and invalid data
                    if i % 3 == 0:
                        # Invalid data
                        data = {
                            'position': [1e8, 1e8, 200000],  # Out of range
                            'velocity': [5000, 0, 0],        # Too high
                            'voltage': 30000                 # Too high
                        }
                    else:
                        # Valid data
                        data = {
                            'position': [i, i*2, i*10],
                            'velocity': [i, i*2, i*3],
                            'voltage': 12000 + i
                        }
                    
                    self.telemetry.update(data)
                    
                    # Check validation results
                    validation_results.append({
                        'worker_id': worker_id,
                        'iteration': i,
                        'errors': len(self.telemetry.validation_errors),
                        'quality': self.telemetry.data_quality,
                        'is_valid': self.telemetry.is_data_valid()
                    })
                    
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Validation worker {worker_id} error: {e}")
        
        # Run validation workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=validation_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify validation results
        assert len(validation_results) == 90  # 3 workers * 30 iterations
        
        # Check that some validation errors were detected
        error_counts = [r['errors'] for r in validation_results]
        assert max(error_counts) > 0  # Should have some validation errors
        
        # Check that quality scores vary appropriately
        quality_scores = [r['quality'] for r in validation_results]
        assert min(quality_scores) < max(quality_scores)  # Should have variation


class TestCircularBufferThreadSafety:
    """Test thread safety of CircularBuffer class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.buffer = CircularBuffer(maxsize=100)
        self.results = []
        self.errors = []
    
    def test_concurrent_buffer_operations(self):
        """Test concurrent buffer operations."""
        def buffer_worker(worker_id: int):
            try:
                for i in range(50):
                    # Add data to buffer
                    data = {'worker_id': worker_id, 'iteration': i, 'timestamp': time.time()}
                    self.buffer.append(data)
                    
                    # Read data occasionally
                    if i % 10 == 0:
                        buffer_data = self.buffer.get_data()
                        self.results.append(len(buffer_data))
                    
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Buffer worker {worker_id} error: {e}")
        
        # Run buffer workers
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=buffer_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify buffer state
        final_data = self.buffer.get_data()
        assert len(final_data) <= 100  # Should not exceed max size
        assert len(final_data) > 0    # Should have some data
        
        # Verify results were recorded
        assert len(self.results) > 0
    
    def test_buffer_overflow_thread_safety(self):
        """Test buffer overflow behavior under concurrent access."""
        def overflow_worker(worker_id: int):
            try:
                for i in range(200):  # More than buffer size
                    data = f"worker_{worker_id}_item_{i}"
                    self.buffer.append(data)
                    
                    if i % 50 == 0:
                        current_size = len(self.buffer.get_data())
                        self.results.append(current_size)
                    
                    time.sleep(0.0001)
            except Exception as e:
                self.errors.append(f"Overflow worker {worker_id} error: {e}")
        
        # Run overflow workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=overflow_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify buffer never exceeded max size
        final_size = len(self.buffer.get_data())
        assert final_size <= 100
        
        # Verify all recorded sizes were within limits
        for size in self.results:
            assert size <= 100
    
    def test_buffer_clear_thread_safety(self):
        """Test buffer clear operation under concurrent access."""
        clear_count = 0
        
        def add_worker():
            try:
                for i in range(100):
                    self.buffer.append(f"item_{i}")
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Add worker error: {e}")
        
        def clear_worker():
            nonlocal clear_count
            try:
                for i in range(10):
                    time.sleep(0.01)  # Let some data accumulate
                    self.buffer.clear()
                    clear_count += 1
                    time.sleep(0.005)
            except Exception as e:
                self.errors.append(f"Clear worker error: {e}")
        
        def read_worker():
            try:
                for i in range(50):
                    data = self.buffer.get_data()
                    self.results.append(len(data))
                    time.sleep(0.002)
            except Exception as e:
                self.errors.append(f"Read worker error: {e}")
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=add_worker),
            threading.Thread(target=clear_worker),
            threading.Thread(target=read_worker)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify clear operations completed
        assert clear_count == 10
        
        # Verify read operations completed
        assert len(self.results) == 50


class TestCombinedThreadSafety:
    """Test thread safety when multiple components interact."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.telemetry_data = TelemetryData()
        self.buffers = [CircularBuffer(maxsize=50) for _ in range(3)]
        self.results = []
        self.errors = []
    
    def test_telemetry_with_multiple_buffers(self):
        """Test telemetry data updates with multiple circular buffers."""
        def telemetry_updater(worker_id: int):
            try:
                for i in range(40):
                    # Update telemetry
                    data = {
                        'position': [worker_id + i, worker_id * 2 + i, worker_id * 3 + i],
                        'velocity': [i, i * 2, i * 3],
                        'voltage': 12000 + i,
                        'timestamp': time.time()
                    }
                    self.telemetry_data.update(data)
                    
                    # Store in buffers
                    for buffer_idx, buffer in enumerate(self.buffers):
                        buffer_data = {
                            'worker_id': worker_id,
                            'iteration': i,
                            'buffer_id': buffer_idx,
                            'telemetry_summary': self.telemetry_data.get_summary()
                        }
                        buffer.append(buffer_data)
                    
                    self.results.append(f"Worker {worker_id} iteration {i}")
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Telemetry updater {worker_id} error: {e}")
        
        def buffer_reader(buffer_idx: int):
            try:
                for i in range(60):
                    data = self.buffers[buffer_idx].get_data()
                    if data:
                        latest = data[-1]
                        self.results.append(f"Buffer {buffer_idx} read: {len(data)} items")
                    time.sleep(0.0015)
            except Exception as e:
                self.errors.append(f"Buffer reader {buffer_idx} error: {e}")
        
        # Run combined operations
        threads = []
        
        # Telemetry updaters
        for worker_id in range(2):
            thread = threading.Thread(target=telemetry_updater, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Buffer readers
        for buffer_idx in range(3):
            thread = threading.Thread(target=buffer_reader, args=(buffer_idx,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        
        # Verify operations completed
        telemetry_updates = [r for r in self.results if "Worker" in r and "iteration" in r]
        buffer_reads = [r for r in self.results if "Buffer" in r and "read" in r]
        
        assert len(telemetry_updates) == 80  # 2 workers * 40 iterations
        assert len(buffer_reads) > 0
        
        # Verify final state
        assert self.telemetry_data.data_quality >= 0.0
        
        for buffer in self.buffers:
            data = buffer.get_data()
            assert len(data) <= 50  # Should not exceed max size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])