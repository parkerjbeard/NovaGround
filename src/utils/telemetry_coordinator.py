"""
Thread-Safe Telemetry Data Coordinator

This module provides a centralized, thread-safe mechanism for sharing telemetry data
between different components of the NovoGround system. It implements the observer pattern
with thread safety guarantees and provides data validation and caching.
"""

import threading
import time
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from PyQt5.QtCore import QObject, pyqtSignal
from src.utils.telemetry_data import TelemetryData
from collections import deque
import weakref


class TelemetryEventType(Enum):
    """Types of telemetry events."""
    DATA_UPDATED = "data_updated"
    CONNECTION_CHANGED = "connection_changed"
    VALIDATION_ERROR = "validation_error"
    DATA_STALE = "data_stale"


@dataclass
class TelemetryEvent:
    """Represents a telemetry event."""
    event_type: TelemetryEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    priority: int = 1  # 1=low, 2=medium, 3=high


class TelemetryObserver:
    """Base class for telemetry observers."""
    
    def on_telemetry_update(self, event: TelemetryEvent) -> None:
        """Called when telemetry data is updated."""
        pass
    
    def on_telemetry_error(self, event: TelemetryEvent) -> None:
        """Called when a telemetry error occurs."""
        pass


class TelemetryCoordinator(QObject):
    """
    Thread-safe telemetry data coordinator that manages data sharing between components.
    
    Features:
    - Thread-safe data access with RLock
    - Observer pattern for component notifications
    - Data validation and quality monitoring
    - Historical data buffering
    - Event-driven architecture with Qt signals
    - Automatic stale data detection
    - Performance monitoring
    """
    
    # Qt signals for cross-thread communication
    telemetry_updated = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(bool, str)
    validation_error = pyqtSignal(str, dict)
    data_stale = pyqtSignal(float)
    
    def __init__(self, max_history: int = 1000, stale_timeout: float = 10.0):
        super().__init__()
        
        # Thread safety
        self._lock = threading.RLock()
        self._observer_lock = threading.RLock()
        
        # Current state
        self._current_telemetry: Optional[TelemetryData] = None
        self._last_update_time = 0.0
        self._update_count = 0
        self._error_count = 0
        self._stale_timeout = stale_timeout
        
        # Data history
        self._max_history = max_history
        self._telemetry_history = deque(maxlen=max_history)
        self._event_history = deque(maxlen=max_history)
        
        # Observers (using weak references to prevent memory leaks)
        self._observers: List[weakref.ReferenceType] = []
        
        # Performance metrics
        self._metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'validation_errors': 0,
            'observer_notifications': 0,
            'average_update_rate': 0.0,
            'last_metrics_reset': time.time()
        }
        
        # Connection state
        self._connected = False
        self._connection_mode = "Disconnected"
        
        logging.info("TelemetryCoordinator initialized")
    
    def register_observer(self, observer: TelemetryObserver) -> bool:
        """
        Register an observer for telemetry updates.
        
        :param observer: Observer instance to register
        :return: True if registered successfully
        """
        try:
            with self._observer_lock:
                # Clean up dead weak references
                self._cleanup_dead_observers()
                
                # Add new observer as weak reference
                weak_observer = weakref.ref(observer)
                self._observers.append(weak_observer)
                
                logging.info(f"Registered telemetry observer: {type(observer).__name__}")
                return True
                
        except Exception as e:
            logging.error(f"Error registering observer: {e}")
            return False
    
    def unregister_observer(self, observer: TelemetryObserver) -> bool:
        """
        Unregister an observer.
        
        :param observer: Observer instance to unregister
        :return: True if unregistered successfully
        """
        try:
            with self._observer_lock:
                # Find and remove the observer
                for weak_ref in self._observers[:]:
                    obs = weak_ref()
                    if obs is observer:
                        self._observers.remove(weak_ref)
                        logging.info(f"Unregistered telemetry observer: {type(observer).__name__}")
                        return True
                
                return False
                
        except Exception as e:
            logging.error(f"Error unregistering observer: {e}")
            return False
    
    def _cleanup_dead_observers(self):
        """Remove dead weak references from observer list."""
        self._observers = [ref for ref in self._observers if ref() is not None]
    
    def update_telemetry(self, telemetry_data: Dict[str, Any], source: str = "unknown") -> bool:
        """
        Thread-safe telemetry data update.
        
        :param telemetry_data: New telemetry data
        :param source: Data source identifier
        :return: True if update was successful
        """
        try:
            with self._lock:
                self._metrics['total_updates'] += 1
                
                # Create or update TelemetryData object
                if self._current_telemetry is None:
                    self._current_telemetry = TelemetryData()
                
                # Update the telemetry data with validation
                self._current_telemetry.update(telemetry_data)
                
                # Check for validation errors
                if self._current_telemetry.validation_errors:
                    self._metrics['validation_errors'] += 1
                    self._handle_validation_errors(self._current_telemetry.validation_errors, source)
                else:
                    self._metrics['successful_updates'] += 1
                
                # Update timing
                current_time = time.time()
                self._last_update_time = current_time
                self._update_count += 1
                
                # Store in history
                history_entry = {
                    'timestamp': current_time,
                    'data': self._current_telemetry.get_safe_copy(),
                    'source': source
                }
                self._telemetry_history.append(history_entry)
                
                # Create event
                event = TelemetryEvent(
                    event_type=TelemetryEventType.DATA_UPDATED,
                    timestamp=current_time,
                    data=self._current_telemetry.to_dict(),
                    source=source,
                    priority=2
                )
                self._event_history.append(event)
                
                # Notify observers
                self._notify_observers(event)
                
                # Emit Qt signal
                self.telemetry_updated.emit(telemetry_data)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                logging.debug(f"Telemetry updated from {source}: quality={self._current_telemetry.data_quality:.2f}")
                return True
                
        except Exception as e:
            self._metrics['total_updates'] += 1  # Count failed attempts
            logging.error(f"Error updating telemetry: {e}")
            self._handle_error(TelemetryEventType.VALIDATION_ERROR, str(e), source)
            return False
    
    def get_current_telemetry(self) -> Optional[TelemetryData]:
        """
        Thread-safe getter for current telemetry data.
        
        :return: Copy of current telemetry data or None
        """
        with self._lock:
            if self._current_telemetry is None:
                return None
            return self._current_telemetry.get_safe_copy()
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current telemetry status.
        
        :return: Summary dictionary
        """
        with self._lock:
            if self._current_telemetry is None:
                return {'status': 'no_data'}
            
            return {
                'status': 'active',
                'last_update': self._last_update_time,
                'age_seconds': time.time() - self._last_update_time,
                'is_fresh': self.is_data_fresh(),
                'data_quality': self._current_telemetry.data_quality,
                'validation_errors': len(self._current_telemetry.validation_errors),
                'update_count': self._update_count,
                'summary': self._current_telemetry.get_summary()
            }
    
    def is_data_fresh(self, timeout: Optional[float] = None) -> bool:
        """
        Check if current telemetry data is fresh.
        
        :param timeout: Custom timeout in seconds (uses default if None)
        :return: True if data is fresh
        """
        with self._lock:
            if self._current_telemetry is None:
                return False
            
            age = time.time() - self._last_update_time
            timeout = timeout or self._stale_timeout
            return age <= timeout
    
    def get_telemetry_history(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get telemetry history.
        
        :param max_entries: Maximum number of entries to return
        :return: List of historical telemetry entries
        """
        with self._lock:
            history = list(self._telemetry_history)
            if max_entries:
                history = history[-max_entries:]
            
            # Convert TelemetryData objects to dictionaries
            serializable_history = []
            for entry in history:
                serializable_entry = entry.copy()
                if isinstance(entry['data'], TelemetryData):
                    serializable_entry['data'] = entry['data'].to_dict()
                serializable_history.append(serializable_entry)
            
            return serializable_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        with self._lock:
            self._metrics = {
                'total_updates': 0,
                'successful_updates': 0,
                'validation_errors': 0,
                'observer_notifications': 0,
                'average_update_rate': 0.0,
                'last_metrics_reset': time.time()
            }
    
    def set_connection_status(self, connected: bool, mode: str = ""):
        """
        Update connection status.
        
        :param connected: Connection status
        :param mode: Connection mode description
        """
        with self._lock:
            old_connected = self._connected
            self._connected = connected
            self._connection_mode = mode
            
            if old_connected != connected:
                event = TelemetryEvent(
                    event_type=TelemetryEventType.CONNECTION_CHANGED,
                    data={'connected': connected, 'mode': mode},
                    priority=3
                )
                self._event_history.append(event)
                self._notify_observers(event)
                self.connection_status_changed.emit(connected, mode)
                
                logging.info(f"Connection status changed: {connected} ({mode})")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        with self._lock:
            return {
                'connected': self._connected,
                'mode': self._connection_mode,
                'last_update': self._last_update_time,
                'data_fresh': self.is_data_fresh()
            }
    
    def check_data_staleness(self):
        """Check for stale data and notify if needed."""
        if not self.is_data_fresh():
            age = time.time() - self._last_update_time
            event = TelemetryEvent(
                event_type=TelemetryEventType.DATA_STALE,
                data={'age_seconds': age},
                priority=2
            )
            self._event_history.append(event)
            self._notify_observers(event)
            self.data_stale.emit(age)
            logging.warning(f"Telemetry data is stale: {age:.1f} seconds old")
    
    def _handle_validation_errors(self, errors: List[str], source: str):
        """Handle validation errors."""
        error_data = {
            'errors': errors,
            'source': source,
            'timestamp': time.time()
        }
        
        event = TelemetryEvent(
            event_type=TelemetryEventType.VALIDATION_ERROR,
            data=error_data,
            source=source,
            priority=2
        )
        self._event_history.append(event)
        self._notify_observers(event)
        self.validation_error.emit(f"Validation errors from {source}", error_data)
        
        logging.warning(f"Telemetry validation errors from {source}: {errors}")
    
    def _handle_error(self, event_type: TelemetryEventType, error_msg: str, source: str):
        """Handle general errors."""
        event = TelemetryEvent(
            event_type=event_type,
            data={'error': error_msg, 'source': source},
            source=source,
            priority=2
        )
        self._event_history.append(event)
        self._notify_observers(event)
    
    def _notify_observers(self, event: TelemetryEvent):
        """Notify all registered observers of an event."""
        with self._observer_lock:
            # Clean up dead references first
            self._cleanup_dead_observers()
            
            for weak_ref in self._observers[:]:
                observer = weak_ref()
                if observer is not None:
                    try:
                        if event.event_type == TelemetryEventType.DATA_UPDATED:
                            observer.on_telemetry_update(event)
                        else:
                            observer.on_telemetry_error(event)
                        
                        self._metrics['observer_notifications'] += 1
                        
                    except Exception as e:
                        logging.error(f"Error notifying observer {type(observer).__name__}: {e}")
                        # Remove problematic observer
                        self._observers.remove(weak_ref)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        current_time = time.time()
        time_since_reset = current_time - self._metrics['last_metrics_reset']
        
        if time_since_reset > 0:
            self._metrics['average_update_rate'] = self._metrics['total_updates'] / time_since_reset
    
    def shutdown(self):
        """Shutdown the coordinator and cleanup resources."""
        with self._lock:
            logging.info("Shutting down TelemetryCoordinator")
            
            # Clear observers
            with self._observer_lock:
                self._observers.clear()
            
            # Clear history
            self._telemetry_history.clear()
            self._event_history.clear()
            
            # Reset state
            self._current_telemetry = None
            self._connected = False
            
            logging.info("TelemetryCoordinator shutdown complete")


# Global telemetry coordinator instance
telemetry_coordinator = TelemetryCoordinator()