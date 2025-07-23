"""
Thread-Safe State Manager

This module provides a centralized, thread-safe state management system for the
NovoGround application. It manages all shared state variables and ensures atomic
operations across multiple threads.
"""

import threading
import time
import logging
from typing import Any, Dict, Optional, Callable, Set, List
from dataclasses import dataclass, field
from enum import Enum
from PyQt5.QtCore import QObject, pyqtSignal
import json
from pathlib import Path


class StateChangeType(Enum):
    """Types of state changes."""
    SET = "set"
    UPDATE = "update" 
    DELETE = "delete"
    RESET = "reset"


@dataclass
class StateChange:
    """Represents a state change event."""
    change_type: StateChangeType
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


class StateManager(QObject):
    """
    Thread-safe state manager for shared application state.
    
    Features:
    - Thread-safe access with RLock
    - State change notifications via Qt signals
    - Change history tracking
    - Atomic state updates
    - State validation
    - Persistent state storage
    - Observer pattern for state changes
    """
    
    # Qt signals for state changes
    state_changed = pyqtSignal(str, object, object)  # key, old_value, new_value
    system_state_changed = pyqtSignal(str)  # system status
    connection_state_changed = pyqtSignal(bool, str)  # connected, mode
    error_state_changed = pyqtSignal(str, str)  # error_type, message
    
    def __init__(self, persistent_file: Optional[str] = None):
        super().__init__()
        
        # Thread safety
        self._lock = threading.RLock()
        self._observer_lock = threading.RLock()
        
        # State storage
        self._state: Dict[str, Any] = {}
        self._default_values: Dict[str, Any] = {}
        self._validators: Dict[str, Callable[[Any], bool]] = {}
        self._observers: Dict[str, Set[Callable]] = {}
        
        # Change tracking
        self._change_history = []
        self._max_history = 1000
        
        # Persistence
        self._persistent_file = persistent_file
        self._auto_persist = True
        
        # Initialize default state
        self._initialize_default_state()
        
        # Load persistent state if file exists
        if self._persistent_file:
            self._load_persistent_state()
        
        logging.info("StateManager initialized")
    
    def _initialize_default_state(self):
        """Initialize default application state."""
        defaults = {
            # Connection state
            'backend_connected': False,
            'connection_mode': 'Disconnected',
            'connection_retries': 0,
            'last_connection_attempt': 0.0,
            
            # System state
            'system_status': 'Initializing',
            'system_armed': False,
            'mission_active': False,
            'mission_phase': 'idle',
            'emergency_stop': False,
            
            # Telemetry state
            'telemetry_active': False,
            'telemetry_rate': 0.0,
            'telemetry_errors': 0,
            'last_telemetry_time': 0.0,
            'telemetry_quality': 0.0,
            
            # UI state
            'main_window_geometry': None,
            'selected_tab': 0,
            'graph_settings': {},
            'alerts_enabled': True,
            
            # Application state
            'app_version': '1.0.0',
            'startup_time': time.time(),
            'shutdown_requested': False,
            'debug_mode': False,
            
            # Performance state
            'frame_rate': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'update_rate': 0.0,
            
            # Error state
            'last_error': None,
            'error_count': 0,
            'warning_count': 0,
            'critical_errors': []
        }
        
        # Set defaults
        for key, value in defaults.items():
            self._default_values[key] = value
            self._state[key] = value
        
        # Set up validators
        self._setup_validators()
    
    def _setup_validators(self):
        """Set up state validators."""
        self._validators.update({
            'backend_connected': lambda x: isinstance(x, bool),
            'system_armed': lambda x: isinstance(x, bool),
            'mission_active': lambda x: isinstance(x, bool),
            'emergency_stop': lambda x: isinstance(x, bool),
            'telemetry_rate': lambda x: isinstance(x, (int, float)) and x >= 0,
            'telemetry_errors': lambda x: isinstance(x, int) and x >= 0,
            'telemetry_quality': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            'frame_rate': lambda x: isinstance(x, (int, float)) and x >= 0,
            'selected_tab': lambda x: isinstance(x, int) and x >= 0,
            'connection_retries': lambda x: isinstance(x, int) and x >= 0,
            'error_count': lambda x: isinstance(x, int) and x >= 0,
            'warning_count': lambda x: isinstance(x, int) and x >= 0,
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Thread-safe getter for state values.
        
        :param key: State key
        :param default: Default value if key doesn't exist
        :return: State value or default
        """
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any, source: str = "unknown") -> bool:
        """
        Thread-safe setter for state values with validation.
        
        :param key: State key
        :param value: New value
        :param source: Source of the change
        :return: True if set successfully
        """
        try:
            with self._lock:
                # Validate value if validator exists
                if key in self._validators:
                    if not self._validators[key](value):
                        logging.warning(f"Invalid value for state key '{key}': {value}")
                        return False
                
                # Get old value
                old_value = self._state.get(key)
                
                # Set new value
                self._state[key] = value
                
                # Track change
                change = StateChange(
                    change_type=StateChangeType.SET,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    source=source
                )
                self._add_change_history(change)
                
                # Notify observers
                self._notify_observers(key, old_value, value)
                
                # Emit signals for special keys
                self._emit_special_signals(key, old_value, value)
                
                # Auto-persist if enabled
                if self._auto_persist and self._persistent_file:
                    self._save_persistent_state()
                
                logging.debug(f"State '{key}' changed from {old_value} to {value} (source: {source})")
                return True
                
        except Exception as e:
            logging.error(f"Error setting state '{key}': {e}")
            return False
    
    def update(self, updates: Dict[str, Any], source: str = "unknown") -> bool:
        """
        Thread-safe atomic update of multiple state values.
        
        :param updates: Dictionary of key-value pairs to update
        :param source: Source of the changes
        :return: True if all updates successful
        """
        try:
            with self._lock:
                # Validate all updates first
                for key, value in updates.items():
                    if key in self._validators:
                        if not self._validators[key](value):
                            logging.warning(f"Invalid value for state key '{key}': {value}")
                            return False
                
                # Apply all updates atomically
                changes = []
                for key, value in updates.items():
                    old_value = self._state.get(key)
                    self._state[key] = value
                    
                    change = StateChange(
                        change_type=StateChangeType.UPDATE,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                        source=source
                    )
                    changes.append(change)
                    self._add_change_history(change)
                    
                    # Notify observers
                    self._notify_observers(key, old_value, value)
                    
                    # Emit signals for special keys
                    self._emit_special_signals(key, old_value, value)
                
                # Auto-persist if enabled
                if self._auto_persist and self._persistent_file:
                    self._save_persistent_state()
                
                logging.debug(f"Updated {len(updates)} state values (source: {source})")
                return True
                
        except Exception as e:
            logging.error(f"Error updating state: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1, source: str = "unknown") -> bool:
        """
        Thread-safe atomic increment of numeric state values.
        
        :param key: State key
        :param amount: Amount to increment (can be negative)
        :param source: Source of the change
        :return: True if incremented successfully
        """
        with self._lock:
            current_value = self._state.get(key, 0)
            if isinstance(current_value, (int, float)):
                return self.set(key, current_value + amount, source)
            else:
                logging.warning(f"Cannot increment non-numeric state '{key}': {current_value}")
                return False
    
    def toggle(self, key: str, source: str = "unknown") -> bool:
        """
        Thread-safe atomic toggle of boolean state values.
        
        :param key: State key
        :param source: Source of the change
        :return: True if toggled successfully
        """
        with self._lock:
            current_value = self._state.get(key, False)
            if isinstance(current_value, bool):
                return self.set(key, not current_value, source)
            else:
                logging.warning(f"Cannot toggle non-boolean state '{key}': {current_value}")
                return False
    
    def reset(self, key: str, source: str = "unknown") -> bool:
        """
        Reset a state value to its default.
        
        :param key: State key to reset
        :param source: Source of the change
        :return: True if reset successfully
        """
        if key in self._default_values:
            return self.set(key, self._default_values[key], source)
        else:
            logging.warning(f"No default value for state key '{key}'")
            return False
    
    def reset_all(self, source: str = "system") -> bool:
        """
        Reset all state values to defaults.
        
        :param source: Source of the change
        :return: True if reset successfully
        """
        try:
            with self._lock:
                for key, default_value in self._default_values.items():
                    old_value = self._state.get(key)
                    self._state[key] = default_value
                    
                    change = StateChange(
                        change_type=StateChangeType.RESET,
                        key=key,
                        old_value=old_value,
                        new_value=default_value,
                        source=source
                    )
                    self._add_change_history(change)
                    
                    # Notify observers
                    self._notify_observers(key, old_value, default_value)
                    
                    # Emit signals for special keys
                    self._emit_special_signals(key, old_value, default_value)
                
                logging.info("All state values reset to defaults")
                return True
                
        except Exception as e:
            logging.error(f"Error resetting state: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all state values."""
        with self._lock:
            return self._state.copy()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state."""
        with self._lock:
            return {
                'total_keys': len(self._state),
                'connection_state': {
                    'connected': self._state.get('backend_connected', False),
                    'mode': self._state.get('connection_mode', 'Unknown'),
                    'retries': self._state.get('connection_retries', 0)
                },
                'system_state': {
                    'status': self._state.get('system_status', 'Unknown'),
                    'armed': self._state.get('system_armed', False),
                    'mission_active': self._state.get('mission_active', False),
                    'emergency_stop': self._state.get('emergency_stop', False)
                },
                'telemetry_state': {
                    'active': self._state.get('telemetry_active', False),
                    'rate': self._state.get('telemetry_rate', 0.0),
                    'errors': self._state.get('telemetry_errors', 0),
                    'quality': self._state.get('telemetry_quality', 0.0)
                },
                'error_state': {
                    'error_count': self._state.get('error_count', 0),
                    'warning_count': self._state.get('warning_count', 0),
                    'last_error': self._state.get('last_error', None)
                },
                'performance_state': {
                    'frame_rate': self._state.get('frame_rate', 0.0),
                    'memory_usage': self._state.get('memory_usage', 0.0),
                    'cpu_usage': self._state.get('cpu_usage', 0.0)
                }
            }
    
    def add_observer(self, key: str, callback: Callable) -> bool:
        """
        Add an observer for state changes.
        
        :param key: State key to observe
        :param callback: Callback function (old_value, new_value) -> None
        :return: True if added successfully
        """
        try:
            with self._observer_lock:
                if key not in self._observers:
                    self._observers[key] = set()
                self._observers[key].add(callback)
                logging.debug(f"Added observer for state key '{key}'")
                return True
        except Exception as e:
            logging.error(f"Error adding observer: {e}")
            return False
    
    def remove_observer(self, key: str, callback: Callable) -> bool:
        """
        Remove an observer for state changes.
        
        :param key: State key
        :param callback: Callback function to remove
        :return: True if removed successfully
        """
        try:
            with self._observer_lock:
                if key in self._observers:
                    self._observers[key].discard(callback)
                    if not self._observers[key]:
                        del self._observers[key]
                    logging.debug(f"Removed observer for state key '{key}'")
                    return True
                return False
        except Exception as e:
            logging.error(f"Error removing observer: {e}")
            return False
    
    def _notify_observers(self, key: str, old_value: Any, new_value: Any):
        """Notify observers of state changes."""
        with self._observer_lock:
            if key in self._observers:
                for callback in self._observers[key].copy():  # Copy to avoid modification during iteration
                    try:
                        callback(old_value, new_value)
                    except Exception as e:
                        logging.error(f"Error in state observer callback: {e}")
                        # Remove problematic callback
                        self._observers[key].discard(callback)
    
    def _emit_special_signals(self, key: str, old_value: Any, new_value: Any):
        """Emit Qt signals for special state keys."""
        try:
            # General state change signal
            self.state_changed.emit(key, old_value, new_value)
            
            # Specific signals
            if key == 'system_status':
                self.system_state_changed.emit(str(new_value))
            elif key in ['backend_connected', 'connection_mode']:
                connected = self._state.get('backend_connected', False)
                mode = self._state.get('connection_mode', 'Unknown')
                self.connection_state_changed.emit(connected, mode)
            elif key == 'last_error' and new_value:
                self.error_state_changed.emit("error", str(new_value))
                
        except Exception as e:
            logging.error(f"Error emitting state signals: {e}")
    
    def _add_change_history(self, change: StateChange):
        """Add a change to the history."""
        self._change_history.append(change)
        
        # Trim history if too long
        if len(self._change_history) > self._max_history:
            self._change_history = self._change_history[-self._max_history:]
    
    def get_change_history(self, max_entries: Optional[int] = None) -> List[StateChange]:
        """Get state change history."""
        with self._lock:
            history = self._change_history.copy()
            if max_entries:
                history = history[-max_entries:]
            return history
    
    def _save_persistent_state(self):
        """Save state to persistent storage."""
        if not self._persistent_file:
            return
        
        try:
            # Only save non-sensitive, persistent state
            persistent_keys = [
                'main_window_geometry', 'selected_tab', 'graph_settings',
                'alerts_enabled', 'debug_mode'
            ]
            
            persistent_state = {
                key: self._state[key] for key in persistent_keys 
                if key in self._state
            }
            
            with open(self._persistent_file, 'w') as f:
                json.dump(persistent_state, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving persistent state: {e}")
    
    def _load_persistent_state(self):
        """Load state from persistent storage."""
        if not self._persistent_file or not Path(self._persistent_file).exists():
            return
        
        try:
            with open(self._persistent_file, 'r') as f:
                persistent_state = json.load(f)
            
            # Update state with persistent values
            for key, value in persistent_state.items():
                if key in self._default_values:  # Only load known keys
                    self._state[key] = value
                    
            logging.info(f"Loaded persistent state from {self._persistent_file}")
            
        except Exception as e:
            logging.error(f"Error loading persistent state: {e}")
    
    def shutdown(self):
        """Shutdown the state manager."""
        with self._lock:
            logging.info("Shutting down StateManager")
            
            # Save final state
            if self._auto_persist and self._persistent_file:
                self._save_persistent_state()
            
            # Clear observers
            with self._observer_lock:
                self._observers.clear()
            
            # Clear history
            self._change_history.clear()
            
            logging.info("StateManager shutdown complete")


# Global state manager instance
state_manager = StateManager(persistent_file="~/.novoground/app_state.json")