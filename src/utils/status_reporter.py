"""
Thread-Safe Status and Error Reporting System

This module provides a centralized, thread-safe system for reporting status updates
and errors throughout the NovoGround application.
"""

import threading
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from PyQt5.QtCore import QObject, pyqtSignal


class StatusLevel(IntEnum):
    """Status level enumeration (higher values = more critical)."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class ErrorType(Enum):
    """Types of errors that can be reported."""
    COMMUNICATION = "communication"
    VALIDATION = "validation"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USER_ACTION = "user_action"
    SYSTEM = "system"


@dataclass
class StatusUpdate:
    """Represents a status update."""
    level: StatusLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    category: str = "general"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorReport:
    """Represents an error report."""
    error_type: ErrorType
    message: str
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    severity: StatusLevel = StatusLevel.ERROR
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class StatusReporter(QObject):
    """
    Thread-safe status and error reporting system.
    """
    
    # Qt signals for cross-thread communication
    status_updated = pyqtSignal(int, str, str, dict)  # level, message, source, data
    error_reported = pyqtSignal(str, str, int, dict)  # error_type, message, severity, context
    critical_error = pyqtSignal(str, str, dict)  # message, source, context
    
    def __init__(self):
        super().__init__()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # History tracking
        self._status_history: List[StatusUpdate] = []
        self._error_history: List[ErrorReport] = []
        self._active_errors: Dict[str, ErrorReport] = {}
        
        # Statistics
        self._stats = {
            'total_status_updates': 0,
            'total_errors': 0,
            'critical_errors': 0,
            'resolved_errors': 0,
            'last_error_time': 0.0,
        }
        
        logging.info("StatusReporter initialized")
    
    def report_status(self, level: StatusLevel, message: str, source: str = "unknown", 
                     category: str = "general", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Report a status update.
        """
        try:
            with self._lock:
                status = StatusUpdate(
                    level=level,
                    message=message,
                    source=source,
                    category=category,
                    data=data or {}
                )
                
                self._status_history.append(status)
                self._stats['total_status_updates'] += 1
                
                # Emit signal
                self.status_updated.emit(
                    int(status.level), 
                    status.message, 
                    status.source, 
                    status.data
                )
                
                # Log the status
                log_level_map = {
                    StatusLevel.DEBUG: logging.DEBUG,
                    StatusLevel.INFO: logging.INFO,
                    StatusLevel.WARNING: logging.WARNING,
                    StatusLevel.ERROR: logging.ERROR,
                    StatusLevel.CRITICAL: logging.CRITICAL
                }
                
                log_level = log_level_map.get(status.level, logging.INFO)
                logging.log(log_level, f"[{status.source}] {status.message}")
                
        except Exception as e:
            logging.error(f"Error reporting status: {e}")
    
    def report_error(self, error_type: ErrorType, message: str, exception: Optional[Exception] = None,
                    source: str = "unknown", severity: StatusLevel = StatusLevel.ERROR,
                    context: Optional[Dict[str, Any]] = None) -> str:
        """
        Report an error.
        """
        try:
            with self._lock:
                # Generate error ID
                error_id = f"{error_type.value}_{source}_{int(time.time())}_{hash(message) % 10000}"
                
                error_report = ErrorReport(
                    error_type=error_type,
                    message=message,
                    exception=exception,
                    source=source,
                    severity=severity,
                    context=context or {}
                )
                
                # Add to active errors
                self._active_errors[error_id] = error_report
                self._error_history.append(error_report)
                
                # Update statistics
                self._stats['total_errors'] += 1
                self._stats['last_error_time'] = error_report.timestamp
                
                if error_report.severity >= StatusLevel.CRITICAL:
                    self._stats['critical_errors'] += 1
                
                # Emit signals
                self.error_reported.emit(
                    error_report.error_type.value,
                    error_report.message,
                    int(error_report.severity),
                    error_report.context
                )
                
                if error_report.severity >= StatusLevel.CRITICAL:
                    self.critical_error.emit(
                        error_report.message,
                        error_report.source,
                        error_report.context
                    )
                
                # Log the error
                log_level_map = {
                    StatusLevel.WARNING: logging.WARNING,
                    StatusLevel.ERROR: logging.ERROR,
                    StatusLevel.CRITICAL: logging.CRITICAL
                }
                
                log_level = log_level_map.get(error_report.severity, logging.ERROR)
                log_message = f"[{error_report.source}] {error_report.error_type.value.upper()}: {error_report.message}"
                
                if error_report.exception:
                    log_message += f" | Exception: {str(error_report.exception)}"
                
                logging.log(log_level, log_message)
                
                return error_id
                
        except Exception as e:
            logging.error(f"Error reporting error: {e}")
            return ""
    
    def resolve_error(self, error_id: str, resolution_notes: str = "") -> bool:
        """
        Mark an error as resolved.
        """
        try:
            with self._lock:
                if error_id in self._active_errors:
                    error_report = self._active_errors[error_id]
                    error_report.resolved = True
                    
                    # Move to resolved
                    del self._active_errors[error_id]
                    self._stats['resolved_errors'] += 1
                    
                    logging.info(f"Error {error_id} resolved: {resolution_notes}")
                    return True
                
                return False
                
        except Exception as e:
            logging.error(f"Error resolving error {error_id}: {e}")
            return False
    
    def get_active_errors(self) -> List[ErrorReport]:
        """Get list of active (unresolved) errors."""
        with self._lock:
            return list(self._active_errors.values())
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error status."""
        with self._lock:
            return {
                'active_errors': len(self._active_errors),
                'total_errors': self._stats['total_errors'],
                'critical_errors': self._stats['critical_errors'],
                'resolved_errors': self._stats['resolved_errors'],
                'last_error_time': self._stats['last_error_time'],
            }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of status updates."""
        with self._lock:
            return {
                'total_status_updates': self._stats['total_status_updates'],
                'recent_status': self._status_history[-5:] if self._status_history else []
            }
    
    def shutdown(self):
        """Shutdown the status reporter."""
        logging.info("StatusReporter shutdown complete")


# Global status reporter instance
status_reporter = StatusReporter()

# Convenience functions
def report_info(message: str, source: str = "unknown", **kwargs):
    """Report an info status."""
    status_reporter.report_status(StatusLevel.INFO, message, source, **kwargs)

def report_warning(message: str, source: str = "unknown", **kwargs):
    """Report a warning status."""
    status_reporter.report_status(StatusLevel.WARNING, message, source, **kwargs)

def report_error(message: str, exception: Optional[Exception] = None, source: str = "unknown", **kwargs):
    """Report an error."""
    return status_reporter.report_error(ErrorType.SOFTWARE, message, exception, source, StatusLevel.ERROR, **kwargs)

def report_critical(message: str, exception: Optional[Exception] = None, source: str = "unknown", **kwargs):
    """Report a critical error."""
    return status_reporter.report_error(ErrorType.SYSTEM, message, exception, source, StatusLevel.CRITICAL, **kwargs)