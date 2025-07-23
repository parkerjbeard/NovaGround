from src.utils.constants import LOG_DIR, TELEMETRY_LOG_FILE, EVENT_LOG_FILE, LOG_LEVEL
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import logging.handlers
import threading
import queue
import os
import gzip
import shutil
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import atexit

class AsyncLogHandler(logging.Handler):
    """
    Asynchronous log handler for high-frequency logging operations.
    Uses a queue and background thread to prevent I/O blocking.
    """
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        self.worker_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.close)
    
    def emit(self, record):
        """
        Add log record to queue for asynchronous processing.
        """
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop oldest record if queue is full
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(record)
            except queue.Empty:
                pass
    
    def _worker(self):
        """
        Background worker thread that processes log records.
        """
        while not self.shutdown_event.is_set():
            try:
                record = self.log_queue.get(timeout=0.1)
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Handle errors in background logging
                print(f"AsyncLogHandler error: {e}")
    
    def close(self):
        """
        Gracefully shutdown the async handler.
        """
        self.shutdown_event.set()
        
        # Process remaining items in queue
        try:
            while True:
                record = self.log_queue.get_nowait()
                self.target_handler.emit(record)
                self.log_queue.task_done()
        except queue.Empty:
            pass
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        self.target_handler.close()
        super().close()

class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that compresses old log files.
    """
    
    def doRollover(self):
        """
        Do a rollover and compress the old log file.
        """
        super().doRollover()
        
        # Compress the rotated file
        old_file = f"{self.baseFilename}.1"
        if os.path.exists(old_file):
            compressed_file = f"{old_file}.gz"
            with open(old_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(old_file)

class PerformanceMonitor:
    """
    Monitors logging performance and system resources.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            'total_logs': 0,
            'telemetry_logs': 0,
            'event_logs': 0,
            'errors': 0,
            'queue_overflows': 0,
            'avg_processing_time': 0.0,
            'start_time': time.time()
        }
    
    def record_log(self, log_type: str, processing_time: float = 0.0):
        """
        Record logging metrics.
        """
        with self.lock:
            self.metrics['total_logs'] += 1
            self.metrics[f'{log_type}_logs'] += 1
            
            # Update average processing time
            if processing_time > 0:
                current_avg = self.metrics['avg_processing_time']
                total_logs = self.metrics['total_logs']
                self.metrics['avg_processing_time'] = (
                    (current_avg * (total_logs - 1) + processing_time) / total_logs
                )
    
    def record_error(self, error_type: str = 'general'):
        """
        Record logging errors.
        """
        with self.lock:
            self.metrics['errors'] += 1
            if error_type == 'queue_overflow':
                self.metrics['queue_overflows'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        with self.lock:
            metrics = self.metrics.copy()
            metrics['uptime'] = time.time() - metrics['start_time']
            metrics['logs_per_second'] = (
                metrics['total_logs'] / metrics['uptime'] if metrics['uptime'] > 0 else 0
            )
            return metrics

class ThreadSafeLogger:
    """
    Thread-safe, high-performance logger with rotation, compression, and monitoring.
    Implements singleton pattern to prevent handler duplication.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.setup_lock = threading.Lock()
        self.performance_monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Logger")
        
        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.telemetry_logger = None
        self.event_logger = None
        self.console_handlers_added = False
        
        # Setup loggers
        self._setup_loggers()
        
        # Register cleanup
        atexit.register(self.shutdown)
    
    def _setup_loggers(self):
        """
        Setup telemetry and event loggers with rotation and async handling.
        """
        with self.setup_lock:
            # Setup telemetry logger
            self.telemetry_logger = self._create_logger(
                'telemetry', 
                TELEMETRY_LOG_FILE,
                max_bytes=100*1024*1024,  # 100MB
                backup_count=10,
                use_async=True
            )
            
            # Setup event logger  
            self.event_logger = self._create_logger(
                'event',
                EVENT_LOG_FILE,
                max_bytes=50*1024*1024,  # 50MB
                backup_count=5,
                use_async=False  # Events are less frequent
            )
    
    def _create_logger(self, name: str, log_file: str, max_bytes: int, 
                      backup_count: int, use_async: bool = False) -> logging.Logger:
        """
        Create a logger with rotation, compression, and optional async handling.
        """
        logger = logging.getLogger(f"novoground.{name}")
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
        
        # Create rotating file handler with compression
        file_path = LOG_DIR / log_file
        file_handler = CompressedRotatingFileHandler(
            filename=str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(threadName)-10s | %(name)-15s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Use async handler for high-frequency logging
        if use_async:
            async_handler = AsyncLogHandler(file_handler, queue_size=50000)
            logger.addHandler(async_handler)
        else:
            logger.addHandler(file_handler)
        
        return logger
    
    def _add_console_handler(self, logger: logging.Logger, name: str):
        """
        Add console handler to logger if not already present.
        """
        # Check if console handler already exists
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and 
            not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        )
        
        if not has_console_handler:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                f'%(asctime)s | {name.upper()} | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
            logger.addHandler(console_handler)
    
    def configure_logging(self) -> None:
        """
        Configure logging with console output.
        Prevents duplicate handler registration.
        """
        with self.setup_lock:
            if self.console_handlers_added:
                return
            
            # Add console handlers for real-time monitoring
            self._add_console_handler(self.telemetry_logger, 'telemetry')
            self._add_console_handler(self.event_logger, 'event')
            
            self.console_handlers_added = True
            self.log_event("Logging system initialized with console output", "INFO")
    
    def log_telemetry(self, data: Dict[str, Any]) -> None:
        """
        Log telemetry data with performance monitoring.
        
        :param data: Dictionary containing telemetry data
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(data, dict):
                self.log_event(f"Invalid telemetry data type: {type(data)}", "WARNING")
                return
            
            # Create structured log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'thread': threading.current_thread().name
            }
            
            # Log as JSON for better parsing
            self.telemetry_logger.info(json.dumps(log_entry, default=str))
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self.performance_monitor.record_log('telemetry', processing_time)
            
        except Exception as e:
            self.performance_monitor.record_error('general')
            self.log_event(f"Error logging telemetry: {e}", "ERROR")
    
    def log_event(self, event: str, level: str = 'INFO') -> None:
        """
        Log system events with thread safety and validation.
        
        :param event: Description of the event
        :param level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not isinstance(event, str):
                event = str(event)
            
            level = level.upper()
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                level = 'INFO'
            
            # Get the appropriate log method
            log_method = getattr(self.event_logger, level.lower())
            
            # Create structured log entry
            log_entry = f"[{threading.current_thread().name}] {event}"
            log_method(log_entry)
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self.performance_monitor.record_log('event', processing_time)
            
        except Exception as e:
            self.performance_monitor.record_error('general')
            # Fallback logging to prevent infinite recursion
            print(f"Logger error: {e} (original event: {event})")
    
    def log_async(self, log_func, *args, **kwargs):
        """
        Submit a logging operation to the thread pool for async execution.
        
        :param log_func: Logging function to execute
        :param args: Arguments for the logging function
        :param kwargs: Keyword arguments for the logging function
        """
        try:
            self.executor.submit(log_func, *args, **kwargs)
        except Exception as e:
            self.performance_monitor.record_error('async')
            self.log_event(f"Async logging error: {e}", "ERROR")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current logging performance metrics.
        
        :return: Dictionary containing performance metrics
        """
        return self.performance_monitor.get_metrics()
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old compressed log files.
        
        :param days_to_keep: Number of days of logs to retain
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = cutoff_date.timestamp()
            
            for log_file in LOG_DIR.glob("*.gz"):
                if log_file.stat().st_mtime < cutoff_timestamp:
                    log_file.unlink()
                    self.log_event(f"Cleaned up old log file: {log_file.name}", "INFO")
                    
        except Exception as e:
            self.log_event(f"Error cleaning up logs: {e}", "ERROR")
    
    def check_disk_space(self) -> Dict[str, Any]:
        """
        Check available disk space for logging.
        
        :return: Dictionary with disk space information
        """
        try:
            stat = shutil.disk_usage(LOG_DIR)
            
            total_gb = stat.total / (1024**3)
            free_gb = stat.free / (1024**3)
            used_gb = (stat.total - stat.free) / (1024**3)
            free_percent = (stat.free / stat.total) * 100
            
            disk_info = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'free_percent': round(free_percent, 2)
            }
            
            # Alert if disk space is low
            if free_percent < 10:
                self.log_event(
                    f"Low disk space warning: {free_percent:.1f}% free ({free_gb:.1f} GB)",
                    "WARNING"
                )
            
            return disk_info
            
        except Exception as e:
            self.log_event(f"Error checking disk space: {e}", "ERROR")
            return {}
    
    def force_flush(self):
        """
        Force flush all log handlers.
        """
        try:
            for handler in self.telemetry_logger.handlers:
                if hasattr(handler, 'flush') and not handler.stream.closed:
                    handler.flush()
            for handler in self.event_logger.handlers:
                if hasattr(handler, 'flush') and not handler.stream.closed:
                    handler.flush()
        except Exception as e:
            # Only print error if it's not about closed files during shutdown
            if "closed file" not in str(e):
                print(f"Error flushing logs: {e}")
    
    def shutdown(self):
        """
        Gracefully shutdown the logging system.
        """
        try:
            self.log_event("Shutting down logging system", "INFO")
            self.force_flush()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Close all handlers
            for handler in self.telemetry_logger.handlers[:]:
                handler.close()
                self.telemetry_logger.removeHandler(handler)
            
            for handler in self.event_logger.handlers[:]:
                handler.close()
                self.event_logger.removeHandler(handler)
                
        except Exception as e:
            print(f"Error during logger shutdown: {e}")

# Global logger instance (singleton)
logger = ThreadSafeLogger()

# Convenience aliases for backward compatibility
class Logger:
    """
    Backward compatibility wrapper for the old Logger class.
    """
    
    def __init__(self):
        # Delegate to the singleton ThreadSafeLogger
        pass
    
    def log_telemetry(self, data: Dict[str, Any]) -> None:
        logger.log_telemetry(data)
    
    def log_event(self, event: str, level: str = 'INFO') -> None:
        logger.log_event(event, level)
    
    def configure_logging(self) -> None:
        logger.configure_logging()
