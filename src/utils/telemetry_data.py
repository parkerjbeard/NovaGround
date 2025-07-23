from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union
from datetime import datetime
import threading
import time
import logging
import numpy as np
from copy import deepcopy


@dataclass
class TelemetryData:
    """
    Thread-safe, validated telemetry data structure with comprehensive safety checks.
    Represents structured telemetry data received from the RocketLink C++ backend.
    """
    position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    orientation: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    velocity: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    acceleration: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 9.81))
    voltage: int = 12000  # millivolts
    status_flags: Dict[str, bool] = field(default_factory=lambda: {
        'motor_failure': False,
        'sensor_error': False,
        'system_health': True,
        'sensor_status': True
    })
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Additional computed fields
    altitude: float = 0.0
    velocity_magnitude: float = 0.0
    acceleration_magnitude: float = 9.81
    
    # Validation and quality fields
    data_quality: float = 1.0  # 0.0 to 1.0
    validation_errors: list = field(default_factory=list)
    last_update_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize thread safety and validation after dataclass creation."""
        self._lock = threading.RLock()
        self._update_computed_fields()
        self._validate_all_fields()

    def update(self, new_data: Dict[str, Any]) -> None:
        """
        Thread-safe update of telemetry data with comprehensive validation.

        :param new_data: Dictionary containing new telemetry data.
        :raises ValueError: If validation fails for critical parameters.
        """
        with self._lock:
            try:
                # Clear previous validation errors
                self.validation_errors.clear()
                
                # Validate and update each field
                for key, value in new_data.items():
                    if hasattr(self, key):
                        validated_value = self._validate_field(key, value)
                        setattr(self, key, validated_value)
                    else:
                        self.validation_errors.append(f"Unknown field: {key}")
                        logging.warning(f"TelemetryData: Unknown field '{key}' ignored")
                
                # Update timestamp and computed fields
                self.last_update_time = time.time()
                self._update_computed_fields()
                
                # Calculate data quality based on validation errors
                self._calculate_data_quality()
                
                # Log critical validation errors
                if self.validation_errors:
                    logging.warning(f"TelemetryData validation errors: {self.validation_errors}")
            
            except Exception as e:
                self.validation_errors.append(f"Update failed: {str(e)}")
                logging.error(f"TelemetryData update error: {e}")
                raise ValueError(f"Telemetry update failed: {e}")
    
    def _validate_field(self, field_name: str, value: Any) -> Any:
        """
        Validate individual field values with type checking and range validation.
        
        :param field_name: Name of the field to validate
        :param value: Value to validate
        :return: Validated and potentially corrected value
        """
        try:
            if field_name == 'position':
                return self._validate_position(value)
            elif field_name == 'orientation':
                return self._validate_orientation(value)
            elif field_name == 'velocity':
                return self._validate_velocity(value)
            elif field_name == 'acceleration':
                return self._validate_acceleration(value)
            elif field_name == 'voltage':
                return self._validate_voltage(value)
            elif field_name == 'status_flags':
                return self._validate_status_flags(value)
            elif field_name == 'timestamp':
                return self._validate_timestamp(value)
            elif field_name == 'altitude':
                return self._validate_altitude(value)
            else:
                # For other fields, do basic type validation
                return value
        
        except (ValueError, TypeError) as e:
            self.validation_errors.append(f"{field_name}: {str(e)}")
            # Return current value if validation fails
            return getattr(self, field_name, None)
    
    def _validate_position(self, value: Any) -> Tuple[float, float, float]:
        """Validate position tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Position must be a 3-element tuple/list")
        
        x, y, z = float(value[0]), float(value[1]), float(value[2])
        
        # Basic range checks (adjust as needed)
        if abs(x) > 1e6 or abs(y) > 1e6:  # 1000 km limit
            raise ValueError(f"Position coordinates out of range: ({x}, {y}, {z})")
        
        if z < -1000 or z > 100000:  # -1km to 100km altitude
            raise ValueError(f"Altitude out of range: {z}m")
        
        return (x, y, z)
    
    def _validate_orientation(self, value: Any) -> Tuple[float, float, float]:
        """Validate orientation tuple (pitch, yaw, roll in degrees)."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Orientation must be a 3-element tuple/list")
        
        pitch, yaw, roll = float(value[0]), float(value[1]), float(value[2])
        
        # Normalize angles to [-180, 180] range
        pitch = ((pitch + 180) % 360) - 180
        yaw = ((yaw + 180) % 360) - 180
        roll = ((roll + 180) % 360) - 180
        
        return (pitch, yaw, roll)
    
    def _validate_velocity(self, value: Any) -> Tuple[float, float, float]:
        """Validate velocity tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Velocity must be a 3-element tuple/list")
        
        vx, vy, vz = float(value[0]), float(value[1]), float(value[2])
        
        # Check for reasonable velocity limits (Mach 5 ≈ 1700 m/s)
        velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        if velocity_magnitude > 2000:  # 2000 m/s limit
            raise ValueError(f"Velocity magnitude out of range: {velocity_magnitude:.1f} m/s")
        
        return (vx, vy, vz)
    
    def _validate_acceleration(self, value: Any) -> Tuple[float, float, float]:
        """Validate acceleration tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Acceleration must be a 3-element tuple/list")
        
        ax, ay, az = float(value[0]), float(value[1]), float(value[2])
        
        # Check for reasonable acceleration limits (20g ≈ 200 m/s²)
        accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        if accel_magnitude > 300:  # 30g limit
            raise ValueError(f"Acceleration magnitude out of range: {accel_magnitude:.1f} m/s²")
        
        return (ax, ay, az)
    
    def _validate_voltage(self, value: Any) -> int:
        """Validate voltage value in millivolts."""
        voltage = int(float(value))
        
        if voltage < 5000 or voltage > 20000:  # 5V to 20V range
            raise ValueError(f"Voltage out of range: {voltage}mV")
        
        return voltage
    
    def _validate_status_flags(self, value: Any) -> Dict[str, bool]:
        """Validate status flags dictionary."""
        if not isinstance(value, dict):
            raise ValueError("Status flags must be a dictionary")
        
        validated_flags = {}
        for key, val in value.items():
            if isinstance(val, bool):
                validated_flags[key] = val
            else:
                # Convert to boolean
                validated_flags[key] = bool(val)
        
        return validated_flags
    
    def _validate_timestamp(self, value: Any) -> datetime:
        """Validate timestamp value."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                # Try other common formats
                try:
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {value}")
        elif isinstance(value, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(value)
        else:
            raise ValueError(f"Invalid timestamp type: {type(value)}")
    
    def _validate_altitude(self, value: Any) -> float:
        """Validate altitude value."""
        altitude = float(value)
        
        if altitude < -1000 or altitude > 100000:  # -1km to 100km
            raise ValueError(f"Altitude out of range: {altitude}m")
        
        return altitude
    
    def _update_computed_fields(self) -> None:
        """Update computed fields based on current data."""
        try:
            # Calculate altitude from position
            self.altitude = self.position[2]
            
            # Calculate velocity magnitude
            self.velocity_magnitude = np.sqrt(
                self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2
            )
            
            # Calculate acceleration magnitude
            self.acceleration_magnitude = np.sqrt(
                self.acceleration[0]**2 + self.acceleration[1]**2 + self.acceleration[2]**2
            )
        
        except Exception as e:
            self.validation_errors.append(f"Computed field update error: {str(e)}")
    
    def _calculate_data_quality(self) -> None:
        """Calculate data quality score based on validation errors and timeliness."""
        try:
            # Start with perfect quality
            quality = 1.0
            
            # Reduce quality for each validation error
            quality -= len(self.validation_errors) * 0.1
            
            # Reduce quality for old data
            age = time.time() - self.last_update_time
            if age > 5.0:  # Data older than 5 seconds
                quality -= min(0.5, age / 10.0)
            
            # Check for critical system flags
            if self.status_flags.get('motor_failure', False):
                quality -= 0.3
            if self.status_flags.get('sensor_error', False):
                quality -= 0.2
            
            # Ensure quality is in valid range
            self.data_quality = max(0.0, min(1.0, quality))
        
        except Exception as e:
            self.data_quality = 0.5  # Default to medium quality if calculation fails
            self.validation_errors.append(f"Data quality calculation error: {str(e)}")
    
    def _validate_all_fields(self) -> None:
        """Validate all current field values."""
        try:
            # Clear errors and validate all fields
            self.validation_errors.clear()
            
            # Validate core fields
            self.position = self._validate_position(self.position)
            self.orientation = self._validate_orientation(self.orientation)
            self.velocity = self._validate_velocity(self.velocity)
            self.acceleration = self._validate_acceleration(self.acceleration)
            self.voltage = self._validate_voltage(self.voltage)
            self.status_flags = self._validate_status_flags(self.status_flags)
            
            # Update computed fields and quality
            self._update_computed_fields()
            self._calculate_data_quality()
        
        except Exception as e:
            self.validation_errors.append(f"Full validation error: {str(e)}")
            logging.error(f"TelemetryData validation failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Thread-safe conversion to dictionary with all fields including computed values.

        :return: Complete dictionary representation of the telemetry data.
        """
        with self._lock:
            return {
                'position': self.position,
                'orientation': self.orientation,
                'velocity': self.velocity,
                'acceleration': self.acceleration,
                'voltage': self.voltage,
                'status_flags': deepcopy(self.status_flags),
                'timestamp': self.timestamp.isoformat(),
                'altitude': self.altitude,
                'velocity_magnitude': self.velocity_magnitude,
                'acceleration_magnitude': self.acceleration_magnitude,
                'data_quality': self.data_quality,
                'validation_errors': list(self.validation_errors),
                'last_update_time': self.last_update_time
            }
    
    def get_safe_copy(self) -> 'TelemetryData':
        """
        Get a thread-safe deep copy of the telemetry data.
        
        :return: Deep copy of this TelemetryData instance
        """
        with self._lock:
            # Create new instance and copy data (avoiding lock deep copy issue)
            new_telemetry = TelemetryData()
            
            # Copy all data fields
            new_telemetry.position = deepcopy(self.position)
            new_telemetry.orientation = deepcopy(self.orientation)
            new_telemetry.velocity = deepcopy(self.velocity)
            new_telemetry.acceleration = deepcopy(self.acceleration)
            new_telemetry.voltage = self.voltage
            new_telemetry.status_flags = deepcopy(self.status_flags)
            new_telemetry.timestamp = deepcopy(self.timestamp)
            new_telemetry.altitude = self.altitude
            new_telemetry.velocity_magnitude = self.velocity_magnitude
            new_telemetry.acceleration_magnitude = self.acceleration_magnitude
            new_telemetry.data_quality = self.data_quality
            new_telemetry.validation_errors = list(self.validation_errors)
            new_telemetry.last_update_time = self.last_update_time
            
            return new_telemetry
    
    def is_data_fresh(self, max_age_seconds: float = 5.0) -> bool:
        """
        Check if telemetry data is fresh (recently updated).
        
        :param max_age_seconds: Maximum age in seconds to consider fresh
        :return: True if data is fresh, False otherwise
        """
        with self._lock:
            age = time.time() - self.last_update_time
            return age <= max_age_seconds
    
    def is_data_valid(self, min_quality: float = 0.7) -> bool:
        """
        Check if telemetry data meets minimum quality standards.
        
        :param min_quality: Minimum quality score (0.0 to 1.0)
        :return: True if data quality is acceptable, False otherwise
        """
        with self._lock:
            return self.data_quality >= min_quality and len(self.validation_errors) == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current telemetry status for monitoring.
        
        :return: Summary dictionary with key metrics
        """
        with self._lock:
            return {
                'altitude': self.altitude,
                'velocity_magnitude': self.velocity_magnitude,
                'acceleration_magnitude': self.acceleration_magnitude,
                'voltage_v': self.voltage / 1000.0,
                'data_quality': self.data_quality,
                'is_fresh': self.is_data_fresh(),
                'is_valid': self.is_data_valid(),
                'error_count': len(self.validation_errors),
                'system_health': self.status_flags.get('system_health', False),
                'age_seconds': time.time() - self.last_update_time
            }
    
    def reset_to_defaults(self) -> None:
        """
        Reset all values to safe defaults.
        """
        with self._lock:
            self.position = (0.0, 0.0, 0.0)
            self.orientation = (0.0, 0.0, 0.0)
            self.velocity = (0.0, 0.0, 0.0)
            self.acceleration = (0.0, 0.0, 9.81)
            self.voltage = 12000
            self.status_flags = {
                'motor_failure': False,
                'sensor_error': False,
                'system_health': True,
                'sensor_status': True
            }
            self.timestamp = datetime.utcnow()
            self.validation_errors.clear()
            self.last_update_time = time.time()
            self._update_computed_fields()
            self._calculate_data_quality()
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"TelemetryData(alt={self.altitude:.1f}m, "
                f"vel={self.velocity_magnitude:.1f}m/s, "
                f"accel={self.acceleration_magnitude:.1f}m/s², "
                f"voltage={self.voltage/1000:.1f}V, "
                f"quality={self.data_quality:.2f}, "
                f"errors={len(self.validation_errors)})")


# Utility functions for telemetry data management

def create_telemetry_from_dict(data: Dict[str, Any]) -> TelemetryData:
    """
    Create a TelemetryData instance from a dictionary with validation.
    
    :param data: Dictionary containing telemetry data
    :return: Validated TelemetryData instance
    """
    try:
        # Create instance with defaults
        telemetry = TelemetryData()
        
        # Update with provided data
        telemetry.update(data)
        
        return telemetry
    
    except Exception as e:
        logging.error(f"Failed to create telemetry from dict: {e}")
        # Return default telemetry with error information
        telemetry = TelemetryData()
        telemetry.validation_errors.append(f"Creation failed: {str(e)}")
        return telemetry

def merge_telemetry_data(primary: TelemetryData, secondary: TelemetryData) -> TelemetryData:
    """
    Merge two telemetry data instances, preferring primary data.
    
    :param primary: Primary telemetry data (preferred)
    :param secondary: Secondary telemetry data (fallback)
    :return: Merged telemetry data
    """
    try:
        # Start with primary data
        merged = primary.get_safe_copy()
        
        # Fill in missing or invalid data from secondary
        if not primary.is_data_valid() and secondary.is_data_valid():
            # Use secondary data if primary is invalid
            secondary_dict = secondary.to_dict()
            merged.update(secondary_dict)
        
        return merged
    
    except Exception as e:
        logging.error(f"Failed to merge telemetry data: {e}")
        return primary.get_safe_copy() if primary.is_data_valid() else secondary.get_safe_copy()