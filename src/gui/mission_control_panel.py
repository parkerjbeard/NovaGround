from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, 
                             QGridLayout, QTabWidget, QTextEdit)
from .styles import BUTTON_STYLE, LABEL_STYLE
from pyqtgraph import PlotWidget, mkPen
from PyQt5.QtCore import pyqtSignal, QTimer
import numpy as np
import time
import threading
from collections import deque
from typing import Dict, Any, Optional
import logging

class CircularBuffer:
    """
    Thread-safe circular buffer for efficient telemetry data storage.
    """
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.data = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def append(self, item):
        """Add item to buffer (thread-safe)."""
        with self.lock:
            self.data.append(item)
    
    def get_data(self):
        """Get copy of current data (thread-safe)."""
        with self.lock:
            return list(self.data)
    
    def clear(self):
        """Clear all data (thread-safe)."""
        with self.lock:
            self.data.clear()

class TelemetryValidator:
    """
    Validates telemetry data for safety and integrity.
    """
    
    # Safety limits
    MAX_ALTITUDE = 50000  # meters
    MAX_VELOCITY = 2000   # m/s
    MAX_ACCELERATION = 200  # m/s^2 (~20g)
    MIN_VOLTAGE = 8000    # millivolts (8V)
    MAX_VOLTAGE = 15000   # millivolts (15V)
    
    @staticmethod
    def validate_telemetry(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize telemetry data.
        
        :param data: Raw telemetry data
        :return: Validated and sanitized data with warnings
        """
        warnings = []
        validated_data = data.copy()
        
        try:
            # Validate altitude
            if 'altitude' in data:
                altitude = float(data['altitude'])
                if altitude < -1000 or altitude > TelemetryValidator.MAX_ALTITUDE:
                    warnings.append(f"Altitude out of range: {altitude}m")
                    validated_data['altitude'] = max(-1000, min(altitude, TelemetryValidator.MAX_ALTITUDE))
            
            # Validate velocity
            if 'velocity' in data:
                if isinstance(data['velocity'], (list, tuple)) and len(data['velocity']) >= 3:
                    velocity_mag = np.linalg.norm(data['velocity'])
                    if velocity_mag > TelemetryValidator.MAX_VELOCITY:
                        warnings.append(f"Velocity magnitude out of range: {velocity_mag:.1f}m/s")
                elif isinstance(data['velocity'], (int, float)):
                    if abs(data['velocity']) > TelemetryValidator.MAX_VELOCITY:
                        warnings.append(f"Velocity out of range: {data['velocity']}m/s")
            
            # Validate acceleration
            if 'acceleration' in data:
                if isinstance(data['acceleration'], (list, tuple)) and len(data['acceleration']) >= 3:
                    accel_mag = np.linalg.norm(data['acceleration'])
                    if accel_mag > TelemetryValidator.MAX_ACCELERATION:
                        warnings.append(f"Acceleration magnitude out of range: {accel_mag:.1f}m/s²")
                elif isinstance(data['acceleration'], (int, float)):
                    if abs(data['acceleration']) > TelemetryValidator.MAX_ACCELERATION:
                        warnings.append(f"Acceleration out of range: {data['acceleration']}m/s²")
            
            # Validate voltage
            if 'voltage' in data:
                voltage = float(data['voltage'])
                if voltage < TelemetryValidator.MIN_VOLTAGE or voltage > TelemetryValidator.MAX_VOLTAGE:
                    warnings.append(f"Voltage out of range: {voltage}mV")
            
            validated_data['validation_warnings'] = warnings
            validated_data['validation_timestamp'] = time.time()
            
        except (ValueError, TypeError, AttributeError) as e:
            warnings.append(f"Data validation error: {e}")
            validated_data['validation_warnings'] = warnings
        
        return validated_data

class MissionControlPanel(QWidget):
    """
    Enhanced MissionControlPanel with thread-safe telemetry handling,
    memory-efficient graphing, and comprehensive data validation.
    """
    # Update signals
    launch_mission = pyqtSignal()
    abort_mission = pyqtSignal()
    arm_system = pyqtSignal()
    disarm_system = pyqtSignal()
    
    # Data update signals for thread safety
    telemetry_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super(MissionControlPanel, self).__init__(parent)
        
        # Configuration
        self.max_data_points = 1000
        self.update_interval = 100  # milliseconds
        
        # Thread-safe data storage
        self.telemetry_buffer = CircularBuffer(self.max_data_points)
        self.graph_data = {
            'altitude': CircularBuffer(self.max_data_points),
            'velocity': CircularBuffer(self.max_data_points),
            'acceleration': CircularBuffer(self.max_data_points)
        }
        
        # Performance tracking
        self.last_update_time = 0
        self.update_count = 0
        self.telemetry_rate = 0.0
        
        # Current telemetry state
        self.current_telemetry = {}
        self.last_valid_telemetry = {}
        
        # Initialize UI
        self.init_ui()
        
        # Connect internal signals for thread safety
        self.telemetry_updated.connect(self._update_ui_safe)
        
        # Setup update timer for smooth UI updates
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self._refresh_graphs)
        self.ui_update_timer.start(self.update_interval)
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Add status label
        self.status_label = QLabel("Status: N/A")
        self.status_label.setStyleSheet(LABEL_STYLE)
        main_layout.addWidget(self.status_label)

        # Create tabs for main and secondary data
        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_main_telemetry_tab(), "Main Telemetry")
        tab_widget.addTab(self.create_secondary_telemetry_tab(), "Secondary Telemetry")
        
        main_layout.addWidget(tab_widget)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        # Add clear data button
        self.clear_button = QPushButton("Clear Data")
        self.clear_button.setStyleSheet(BUTTON_STYLE)
        self.clear_button.clicked.connect(self.clear_graphs)
        control_layout.addWidget(self.clear_button)
        
        # Arm Button
        self.arm_button = QPushButton("Arm System")
        self.arm_button.setStyleSheet(BUTTON_STYLE)
        self.arm_button.clicked.connect(self.arm_system.emit)
        control_layout.addWidget(self.arm_button)
        
        # Disarm Button
        self.disarm_button = QPushButton("Disarm System")
        self.disarm_button.setStyleSheet(BUTTON_STYLE)
        self.disarm_button.clicked.connect(self.disarm_system.emit)
        control_layout.addWidget(self.disarm_button)
        
        # Launch Button
        self.launch_button = QPushButton("Launch Mission")
        self.launch_button.setStyleSheet(BUTTON_STYLE)
        self.launch_button.clicked.connect(self.launch_mission.emit)
        control_layout.addWidget(self.launch_button)
        
        # Abort Button
        self.abort_button = QPushButton("Abort Mission")
        self.abort_button.setStyleSheet(BUTTON_STYLE + "background-color: #ff4444;")
        self.abort_button.clicked.connect(self.abort_mission.emit)
        control_layout.addWidget(self.abort_button)
        
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)
    
    def create_main_telemetry_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Telemetry grid
        telemetry_grid = QGridLayout()
        
        # Flight Phase Status
        self.flight_phase_label = QLabel("Flight Phase: N/A")
        self.flight_phase_label.setStyleSheet(LABEL_STYLE)
        telemetry_grid.addWidget(self.flight_phase_label, 0, 0)
        
        # Motor Status
        self.motor_status_label = QLabel("Motor Status: N/A")
        self.motor_status_label.setStyleSheet(LABEL_STYLE)
        telemetry_grid.addWidget(self.motor_status_label, 0, 1)
        
        # GPS Position
        self.gps_position_label = QLabel("GPS: N/A")
        self.gps_position_label.setStyleSheet(LABEL_STYLE)
        telemetry_grid.addWidget(self.gps_position_label, 1, 0)
        
        # Battery Voltage
        self.battery_voltage_label = QLabel("Battery: N/A")
        self.battery_voltage_label.setStyleSheet(LABEL_STYLE)
        telemetry_grid.addWidget(self.battery_voltage_label, 1, 1)
        
        layout.addLayout(telemetry_grid)
        
        # Graphs
        self.altitude_graph = self.create_graph("Altitude (m)")
        self.velocity_graph = self.create_graph("Vertical Velocity (m/s)")
        self.acceleration_graph = self.create_graph("Acceleration (G)")
        
        layout.addWidget(self.altitude_graph)
        layout.addWidget(self.velocity_graph)
        layout.addWidget(self.acceleration_graph)
        
        tab.setLayout(layout)
        return tab

    def create_secondary_telemetry_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Secondary telemetry grid
        secondary_grid = QGridLayout()

        self.rate_of_climb_label = QLabel("Rate of Climb/Descent: N/A")
        secondary_grid.addWidget(self.rate_of_climb_label, 0, 0)

        self.barometric_altitude_label = QLabel("Barometric Altitude: N/A")
        secondary_grid.addWidget(self.barometric_altitude_label, 0, 1)

        self.telemetry_latency_label = QLabel("Telemetry Latency: N/A")
        secondary_grid.addWidget(self.telemetry_latency_label, 1, 0)

        self.power_metrics_label = QLabel("Power Metrics: N/A")
        secondary_grid.addWidget(self.power_metrics_label, 1, 1)

        self.radio_metrics_label = QLabel("Radio Metrics: N/A")
        secondary_grid.addWidget(self.radio_metrics_label, 2, 0)

        layout.addLayout(secondary_grid)

        # System Logs
        self.system_logs = QTextEdit()
        self.system_logs.setReadOnly(True)
        layout.addWidget(QLabel("System Logs:"))
        layout.addWidget(self.system_logs)

        tab.setLayout(layout)
        return tab

    def create_graph(self, title):
        graph = PlotWidget(title=title)
        graph.setBackground('w')
        graph.setLabel('left', title)
        graph.setLabel('bottom', 'Time (s)')
        graph.showGrid(x=True, y=True)
        return graph
    
    def update_status(self, status: str) -> None:
        """
        Updates the flight status indicator with validation.
        
        Args:
            status (str): Current mission status.
        """
        try:
            if not isinstance(status, str):
                status = str(status)
            
            # Color coding for different statuses
            color_map = {
                'launching': 'color: orange; font-weight: bold;',
                'armed': 'color: red; font-weight: bold;',
                'aborting': 'color: red; font-weight: bold;',
                'landed': 'color: green;',
                'idle': 'color: gray;',
                'error': 'color: red; font-weight: bold;'
            }
            
            status_lower = status.lower()
            color_style = color_map.get(status_lower, '')
            
            self.status_label.setText(f"Status: {status}")
            self.status_label.setStyleSheet(LABEL_STYLE + color_style)
            
        except Exception as e:
            logging.error(f"Error updating status: {e}")
            self.status_label.setText("Status: Error")
            self.status_label.setStyleSheet(LABEL_STYLE + 'color: red;')
    
    def update_connectivity(self, connected: bool) -> None:
        """
        Updates the connectivity indicator with enhanced status.
        
        Args:
            connected (bool): Connectivity status.
        """
        try:
            status = "Connected" if connected else "Disconnected"
            color = "color: green;" if connected else "color: red;"
            
            # Add telemetry rate information if available
            if connected and self.telemetry_rate > 0:
                status += f" ({self.telemetry_rate:.1f} Hz)"
            
            if hasattr(self, 'connectivity_label'):
                self.connectivity_label.setText(f"Connectivity: {status}")
                self.connectivity_label.setStyleSheet(LABEL_STYLE + color)
        
        except Exception as e:
            logging.error(f"Error updating connectivity: {e}")
    
    def update_telemetry(self, data: Dict[str, Any]) -> None:
        """
        Main telemetry update method with validation and thread safety.
        
        :param data: Telemetry data dictionary
        """
        try:
            # Validate input data
            if not isinstance(data, dict):
                logging.warning(f"Invalid telemetry data type: {type(data)}")
                return
            
            # Validate and sanitize data
            validated_data = TelemetryValidator.validate_telemetry(data)
            
            # Log validation warnings
            if validated_data.get('validation_warnings'):
                for warning in validated_data['validation_warnings']:
                    logging.warning(f"Telemetry validation: {warning}")
            
            # Update performance metrics
            current_time = time.time()
            if self.last_update_time > 0:
                time_diff = current_time - self.last_update_time
                if time_diff > 0:
                    self.telemetry_rate = 0.9 * self.telemetry_rate + 0.1 * (1.0 / time_diff)
            
            self.last_update_time = current_time
            self.update_count += 1
            
            # Store validated data
            self.current_telemetry = validated_data
            self.telemetry_buffer.append(validated_data)
            
            # Update graph data buffers
            timestamp = validated_data.get('timestamp', current_time)
            
            if 'altitude' in validated_data:
                self.graph_data['altitude'].append((timestamp, validated_data['altitude']))
            
            if 'velocity' in validated_data:
                # Handle both scalar and vector velocity
                velocity = validated_data['velocity']
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                    velocity_magnitude = np.linalg.norm(velocity)
                    self.graph_data['velocity'].append((timestamp, velocity_magnitude))
                elif isinstance(velocity, (int, float)):
                    self.graph_data['velocity'].append((timestamp, abs(velocity)))
            
            if 'acceleration' in validated_data:
                # Handle both scalar and vector acceleration
                acceleration = validated_data['acceleration']
                if isinstance(acceleration, (list, tuple)) and len(acceleration) >= 3:
                    accel_magnitude = np.linalg.norm(acceleration)
                    self.graph_data['acceleration'].append((timestamp, accel_magnitude))
                elif isinstance(acceleration, (int, float)):
                    self.graph_data['acceleration'].append((timestamp, abs(acceleration)))
            
            # Emit signal for thread-safe UI update
            self.telemetry_updated.emit(validated_data)
            
            # Store as last valid telemetry for fallback
            self.last_valid_telemetry = validated_data.copy()
            
        except Exception as e:
            logging.error(f"Error updating telemetry: {e}")
            # Use last valid telemetry as fallback
            if self.last_valid_telemetry:
                self.telemetry_updated.emit(self.last_valid_telemetry)
    
    def _update_ui_safe(self, data: Dict[str, Any]) -> None:
        """
        Thread-safe UI update method called via Qt signal.
        
        :param data: Validated telemetry data
        """
        try:
            # Update main telemetry labels
            self._update_main_telemetry_labels(data)
            
            # Update secondary telemetry
            self._update_secondary_telemetry_labels(data)
            
            # Update system logs
            if 'system_logs' in data:
                self.system_logs.append(str(data['system_logs']))
            
            # Show validation warnings in system logs
            if data.get('validation_warnings'):
                for warning in data['validation_warnings']:
                    self.system_logs.append(f"WARNING: {warning}")
        
        except Exception as e:
            logging.error(f"Error updating UI: {e}")
    
    def _update_main_telemetry_labels(self, data: Dict[str, Any]) -> None:
        """
        Update main telemetry display labels.
        
        :param data: Telemetry data
        """
        try:
            # Flight phase (derive from mission state or altitude)
            if 'mission_state' in data:
                flight_phase = data['mission_state'].title()
            elif 'altitude' in data:
                altitude = data['altitude']
                if altitude < 10:
                    flight_phase = "Ground"
                elif altitude < 100:
                    flight_phase = "Ascending"
                elif data.get('velocity', [0, 0, 0])[2] < 0:  # Negative vertical velocity
                    flight_phase = "Descending"
                else:
                    flight_phase = "Flight"
            else:
                flight_phase = "Unknown"
            
            self.flight_phase_label.setText(f"Flight Phase: {flight_phase}")
            
            # Motor status (derive from status flags or mission state)
            if 'status_flags' in data and 'motor_failure' in data['status_flags']:
                if data['status_flags']['motor_failure']:
                    motor_status = "FAILURE"
                else:
                    motor_status = "Nominal"
            elif 'mission_state' in data:
                if data['mission_state'] == 'launching':
                    motor_status = "Burning"
                elif data['mission_state'] in ['ascending', 'descending']:
                    motor_status = "Burnout"
                else:
                    motor_status = "Idle"
            else:
                motor_status = "Unknown"
            
            self.motor_status_label.setText(f"Motor Status: {motor_status}")
            
            # GPS position
            if 'position' in data:
                position = data['position']
                if isinstance(position, (list, tuple)) and len(position) >= 3:
                    lat, lon, alt = position[0], position[1], position[2]
                    self.gps_position_label.setText(f"GPS: {lat:.6f}, {lon:.6f}, {alt:.1f}m")
                else:
                    self.gps_position_label.setText("GPS: Invalid")
            else:
                self.gps_position_label.setText("GPS: No Data")
            
            # Battery voltage
            if 'voltage' in data:
                voltage = data['voltage']
                voltage_v = voltage / 1000.0  # Convert from millivolts to volts
                
                # Color coding based on voltage level
                if voltage_v < 9.0:
                    color = "color: red;"
                elif voltage_v < 10.5:
                    color = "color: orange;"
                else:
                    color = "color: green;"
                
                self.battery_voltage_label.setText(f"Battery: {voltage_v:.1f}V")
                self.battery_voltage_label.setStyleSheet(LABEL_STYLE + color)
            else:
                self.battery_voltage_label.setText("Battery: No Data")
                self.battery_voltage_label.setStyleSheet(LABEL_STYLE)
        
        except Exception as e:
            logging.error(f"Error updating main telemetry labels: {e}")
    
    def _update_secondary_telemetry_labels(self, data: Dict[str, Any]) -> None:
        """
        Update secondary telemetry display labels.
        
        :param data: Telemetry data
        """
        try:
            # Calculate rate of climb from velocity
            rate_of_climb = "N/A"
            if 'velocity' in data:
                velocity = data['velocity']
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                    rate_of_climb = f"{velocity[2]:.1f}"
                elif isinstance(velocity, (int, float)):
                    rate_of_climb = f"{velocity:.1f}"
            
            self.rate_of_climb_label.setText(f"Rate of Climb/Descent: {rate_of_climb} m/s")
            
            # Barometric altitude (same as altitude if available)
            barometric_altitude = data.get('altitude', 'N/A')
            if isinstance(barometric_altitude, (int, float)):
                barometric_altitude = f"{barometric_altitude:.1f}"
            self.barometric_altitude_label.setText(f"Barometric Altitude: {barometric_altitude} m")
            
            # Telemetry latency (calculate from timestamp)
            telemetry_latency = "N/A"
            if 'timestamp' in data:
                try:
                    if isinstance(data['timestamp'], str):
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        latency = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds() * 1000
                    else:
                        latency = (time.time() - data['timestamp']) * 1000
                    
                    if latency >= 0 and latency < 10000:  # Reasonable range
                        telemetry_latency = f"{latency:.0f}"
                except (ValueError, TypeError, AttributeError):
                    pass
            
            self.telemetry_latency_label.setText(f"Telemetry Latency: {telemetry_latency} ms")
            
            # Power metrics (derive from voltage and system status)
            power_metrics = "N/A"
            if 'voltage' in data:
                voltage_v = data['voltage'] / 1000.0
                if voltage_v > 12.0:
                    power_metrics = "Excellent"
                elif voltage_v > 10.5:
                    power_metrics = "Good"
                elif voltage_v > 9.0:
                    power_metrics = "Low"
                else:
                    power_metrics = "Critical"
            
            self.power_metrics_label.setText(f"Power Metrics: {power_metrics}")
            
            # Radio metrics (derive from telemetry rate and data quality)
            radio_metrics = "N/A"
            if self.telemetry_rate > 0:
                if self.telemetry_rate > 5.0:
                    radio_metrics = "Excellent"
                elif self.telemetry_rate > 1.0:
                    radio_metrics = "Good"
                elif self.telemetry_rate > 0.1:
                    radio_metrics = "Poor"
                else:
                    radio_metrics = "Weak"
                
                radio_metrics += f" ({self.telemetry_rate:.1f} Hz)"
            
            self.radio_metrics_label.setText(f"Radio Metrics: {radio_metrics}")
        
        except Exception as e:
            logging.error(f"Error updating secondary telemetry labels: {e}")
    
    def _refresh_graphs(self) -> None:
        """
        Refresh all graphs with latest data (called by timer).
        """
        try:
            self._update_graph_safe(self.altitude_graph, 'altitude', 'Altitude (m)')
            self._update_graph_safe(self.velocity_graph, 'velocity', 'Velocity (m/s)')
            self._update_graph_safe(self.acceleration_graph, 'acceleration', 'Acceleration (m/s²)')
        except Exception as e:
            logging.error(f"Error refreshing graphs: {e}")
    
    def _update_graph_safe(self, graph: PlotWidget, data_key: str, title: str) -> None:
        """
        Thread-safe graph update with memory management.
        
        :param graph: PlotWidget to update
        :param data_key: Key for data in graph_data dictionary
        :param title: Graph title for display
        """
        try:
            if data_key not in self.graph_data:
                return
            
            # Get current data
            data_points = self.graph_data[data_key].get_data()
            
            if not data_points:
                return
            
            # Extract timestamps and values
            timestamps = [point[0] for point in data_points]
            values = [point[1] for point in data_points]
            
            if not timestamps or not values:
                return
            
            # Convert to numpy arrays for efficiency
            x_data = np.array(timestamps)
            y_data = np.array(values)
            
            # Initialize graph if needed
            if not hasattr(graph, 'data_line'):
                graph.data_line = graph.plot(pen=mkPen('b', width=2))
                graph.setLabel('left', title)
                graph.setLabel('bottom', 'Time (s)')
                graph.showGrid(x=True, y=True)
            
            # Update plot data efficiently
            graph.data_line.setData(x_data, y_data)
            
            # Auto-scale axes
            if len(x_data) > 1:
                graph.setXRange(x_data[0], x_data[-1])
            
            if len(y_data) > 0:
                y_min, y_max = np.min(y_data), np.max(y_data)
                y_range = y_max - y_min
                if y_range > 0:
                    graph.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        except Exception as e:
            logging.error(f"Error updating graph {data_key}: {e}")

    def clear_graphs(self) -> None:
        """
        Clear all graph data and reset displays.
        """
        try:
            for data_buffer in self.graph_data.values():
                data_buffer.clear()
            
            # Clear visual graphs
            for graph in [self.altitude_graph, self.velocity_graph, self.acceleration_graph]:
                if hasattr(graph, 'data_line'):
                    graph.data_line.setData([], [])
        
        except Exception as e:
            logging.error(f"Error clearing graphs: {e}")
    
    def get_telemetry_statistics(self) -> Dict[str, Any]:
        """
        Get current telemetry statistics and performance metrics.
        
        :return: Dictionary with telemetry statistics
        """
        try:
            stats = {
                'update_count': self.update_count,
                'telemetry_rate_hz': self.telemetry_rate,
                'last_update_time': self.last_update_time,
                'buffer_size': len(self.telemetry_buffer.get_data()),
                'graph_data_points': {
                    key: len(buffer.get_data()) 
                    for key, buffer in self.graph_data.items()
                }
            }
            
            # Add current telemetry summary
            if self.current_telemetry:
                stats['current_telemetry'] = {
                    'altitude': self.current_telemetry.get('altitude', 'N/A'),
                    'velocity_magnitude': (
                        np.linalg.norm(self.current_telemetry['velocity']) 
                        if 'velocity' in self.current_telemetry and 
                        isinstance(self.current_telemetry['velocity'], (list, tuple))
                        else self.current_telemetry.get('velocity', 'N/A')
                    ),
                    'voltage': self.current_telemetry.get('voltage', 'N/A'),
                    'has_warnings': bool(self.current_telemetry.get('validation_warnings'))
                }
            
            return stats
        
        except Exception as e:
            logging.error(f"Error getting telemetry statistics: {e}")
            return {'error': str(e)}
    
    def closeEvent(self, event):
        """
        Clean up resources when panel is closed.
        """
        try:
            if hasattr(self, 'ui_update_timer'):
                self.ui_update_timer.stop()
            
            # Clear all data
            self.telemetry_buffer.clear()
            for buffer in self.graph_data.values():
                buffer.clear()
            
            super().closeEvent(event)
        
        except Exception as e:
            logging.error(f"Error during panel cleanup: {e}")