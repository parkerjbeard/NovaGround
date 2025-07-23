from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QAction, QMessageBox, QStatusBar
from .mission_control_panel import MissionControlPanel
from src.panda3d_render.rocket_view import RocketView
from panda3d.core import loadPrcFileData
from .styles import MAIN_WINDOW_STYLE
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QMutex, QMutexLocker
from src.utils.communication import rocket_link, CommunicationError
from src.backend.cplusplus_bindings import Command
from src.utils.telemetry_data import TelemetryData
import logging
import threading
from typing import Dict, Any, Optional

class MainWindow(QMainWindow):
    """
    Thread-safe MainWindow class inheriting from QMainWindow.
    Combines all UI components including rocket view, telemetry dashboard, and mission control panel.
    Features comprehensive thread safety for cross-thread operations.
    """
    
    # Thread-safe signals for cross-thread communication
    telemetry_received = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(bool, str)
    system_status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)  # title, message
    rocket_position_updated = pyqtSignal(tuple, tuple)  # position, orientation
    def __init__(self, rocket_model_path: str = None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("NovoGround Ground Control System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        
        # Thread safety infrastructure
        self._state_mutex = QMutex()
        self._telemetry_mutex = QMutex()
        
        # Thread-safe state variables
        self._backend_connected = False
        self._system_status = "Initializing"
        self._connection_mode = "Disconnected"
        self._last_telemetry = None
        self._telemetry_errors = 0
        self._shutting_down = False
        
        # Configure Panda3D to use a child window
        loadPrcFileData("", "window-type none")
        
        # Initialize backend connection in thread-safe manner
        self.init_backend_connection()
        
        self.init_ui()
        self.init_menu()
        self.show()
        
        # Initialize RocketView after the event loop has processed pending events
        QTimer.singleShot(0, lambda: self.initialize_rocket_view(model_path=rocket_model_path))

        # Set up a timer to update the 3D view
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_3d_view)
        self.render_timer.start(16)  # ~60 FPS
        
        # Connect thread-safe signals
        self._connect_thread_safe_signals()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left pane: Rocket View
        self.rocket_view_widget = QWidget()
        self.rocket_view_widget.setMinimumSize(800, 600)  # Ensure the widget has a size
        main_layout.addWidget(self.rocket_view_widget, 2)
        
        # Right pane: Dashboard and Controls
        right_pane = QVBoxLayout()
        
        # Mission Control Panel (now includes all telemetry)
        self.mission_control_panel = MissionControlPanel()
        right_pane.addWidget(self.mission_control_panel)
        
        main_layout.addLayout(right_pane, 1)
        
        central_widget.setLayout(main_layout)
        
        # Connect signals from mission control to backend commands
        self.mission_control_panel.launch_mission.connect(self.launch_mission)
        self.mission_control_panel.abort_mission.connect(self.abort_mission)
        self.mission_control_panel.arm_system.connect(self.arm_system)
        self.mission_control_panel.disarm_system.connect(self.disarm_system)
        
        # Set up a timer to update telemetry periodically
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(1000)  # Update every 1000 ms (1 second)
        
        # Connection status monitoring timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_connection_status)
        self.status_timer.start(5000)  # Check every 5 seconds
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_connection_status()
        
    def _connect_thread_safe_signals(self):
        """Connect thread-safe signals to their handlers."""
        self.telemetry_received.connect(self._handle_telemetry_update)
        self.connection_status_changed.connect(self._handle_connection_status_change)
        self.system_status_changed.connect(self._handle_system_status_change)
        self.error_occurred.connect(self._handle_error_display)
        self.rocket_position_updated.connect(self._handle_rocket_position_update)
    
    def get_backend_connected(self) -> bool:
        """Thread-safe getter for backend connection status."""
        with QMutexLocker(self._state_mutex):
            return self._backend_connected
    
    def set_backend_connected(self, connected: bool, mode: str = "") -> None:
        """Thread-safe setter for backend connection status."""
        with QMutexLocker(self._state_mutex):
            old_connected = self._backend_connected
            self._backend_connected = connected
            if mode:
                self._connection_mode = mode
            
            # Emit signal if status changed
            if old_connected != connected:
                self.connection_status_changed.emit(connected, self._connection_mode)
    
    def get_system_status(self) -> str:
        """Thread-safe getter for system status."""
        with QMutexLocker(self._state_mutex):
            return self._system_status
    
    def set_system_status(self, status: str) -> None:
        """Thread-safe setter for system status."""
        with QMutexLocker(self._state_mutex):
            if self._system_status != status:
                self._system_status = status
                self.system_status_changed.emit(status)
    
    def get_last_telemetry(self) -> Optional[Dict[str, Any]]:
        """Thread-safe getter for last telemetry data."""
        with QMutexLocker(self._telemetry_mutex):
            return self._last_telemetry.copy() if self._last_telemetry else None
    
    def set_last_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Thread-safe setter for last telemetry data."""
        with QMutexLocker(self._telemetry_mutex):
            self._last_telemetry = telemetry.copy()
            self.telemetry_received.emit(telemetry)

    def initialize_rocket_view(self, model_path: str = None):
        """
        Initializes the RocketView after the main window has been fully shown.
        This ensures that the parent widget's window handle is valid.
        """
        try:
            parent_handle = int(self.rocket_view_widget.winId())
            logging.info(f"Initializing RocketView with parent_handle: {parent_handle}")
            
            self.rocket_view = RocketView(parent_handle=parent_handle)
            logging.info("RocketView initialized successfully")
            
            # Force an update of the 3D view
            self.rocket_view.taskMgr.step()
            logging.info("Initial render step completed")
        except Exception as e:
            logging.exception("Failed to initialize RocketView.")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize RocketView:\n{e}")
            self.close()

    def update_3d_view(self):
        """Thread-safe 3D view update."""
        try:
            if hasattr(self, 'rocket_view') and self.rocket_view:
                # Skip if shutting down
                with QMutexLocker(self._state_mutex):
                    if self._shutting_down:
                        return
                
                self.rocket_view.taskMgr.step()
        except Exception as e:
            logging.error(f"Error updating 3D view: {e}")

    def init_menu(self):
        # Create menu bar
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('File')
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help Menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        QMessageBox.about(self, "About NovoGround GCS",
                          "NovoGround Ground Control System\nVersion 1.0\nDeveloped with PyQt and Panda3D.")
    
    def launch_mission(self):
        """
        Thread-safe launch mission handler.
        Sends launch command to the backend.
        """
        if not self.get_backend_connected():
            self.error_occurred.emit("Connection Error", "Backend not connected. Running in simulation mode.")
            return
            
        try:
            # Execute command in thread-safe manner
            success = self._execute_backend_command(Command.START_MISSION)
            if success:
                self.set_system_status("Launching")
                QMessageBox.information(self, "Mission Control", "Mission Launched.")
            else:
                self.error_occurred.emit("Mission Control", "Failed to launch mission")
        except Exception as e:
            self.error_occurred.emit("Communication Error", f"Failed to launch mission: {e}")
            logging.error(f"Failed to launch mission: {e}")

    def abort_mission(self):
        """
        Thread-safe abort mission handler.
        Sends abort command to the backend.
        """
        if not self.get_backend_connected():
            self.set_system_status("Idle")
            QMessageBox.information(self, "Mission Control", "Mission Aborted (Simulation).")
            return
            
        try:
            # Execute command in thread-safe manner
            success = self._execute_backend_command(Command.ABORT_MISSION)
            if success:
                self.set_system_status("Aborting")
                QMessageBox.warning(self, "Mission Control", "Mission Aborted!")
            else:
                self.error_occurred.emit("Mission Control", "Failed to abort mission")
        except Exception as e:
            self.error_occurred.emit("Communication Error", f"Failed to abort mission: {e}")
            logging.error(f"Failed to abort mission: {e}")

    def arm_system(self):
        """
        Thread-safe arm system handler.
        Sends arm command to the backend.
        """
        try:
            # In a real system, this would send an ARM command
            self.set_system_status("Armed")
            QMessageBox.information(self, "Mission Control", "System Armed.")
        except Exception as e:
            self.error_occurred.emit("Mission Control", f"Failed to arm system: {e}")
            logging.error(f"Failed to arm system: {e}")

    def disarm_system(self):
        """
        Thread-safe disarm system handler.
        Sends disarm command to the backend.
        """
        try:
            # In a real system, this would send a DISARM command
            self.set_system_status("Disarmed")
            QMessageBox.information(self, "Mission Control", "System Disarmed.")
        except Exception as e:
            self.error_occurred.emit("Mission Control", f"Failed to disarm system: {e}")
            logging.error(f"Failed to disarm system: {e}")
    
    def _execute_backend_command(self, command: Command) -> bool:
        """
        Thread-safe backend command execution.
        
        :param command: Command to execute
        :return: True if successful, False otherwise
        """
        try:
            return rocket_link.send_command(command)
        except CommunicationError as e:
            logging.error(f"Backend command failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error executing command: {e}")
            return False
    
    def update_telemetry(self):
        """
        Thread-safe telemetry update handler.
        Fetches telemetry data and updates GUI through signals.
        """
        # Skip if shutting down
        with QMutexLocker(self._state_mutex):
            if self._shutting_down:
                return
        
        try:
            # Fetch telemetry data from backend
            telemetry_data = rocket_link.receive_telemetry()
            
            if telemetry_data:
                # Transform data for GUI consumption
                processed_data = {
                    'altitude': telemetry_data.get('altitude', 0),
                    'velocity': telemetry_data.get('velocity', [0, 0, 0]),
                    'acceleration': telemetry_data.get('acceleration', [0, 0, 0]),
                    'position': tuple(telemetry_data.get('position', [0, 0, 0])),
                    'orientation': tuple(telemetry_data.get('orientation', [0, 0, 0])),
                    'voltage': telemetry_data.get('voltage', 12000),
                    'status_flags': telemetry_data.get('status_flags', {}),
                    'timestamp': telemetry_data.get('timestamp', 0)
                }
                
                # Store and emit through thread-safe signal
                self.set_last_telemetry(processed_data)
                
                # Emit position update for 3D view
                self.rocket_position_updated.emit(
                    processed_data['position'],
                    processed_data['orientation']
                )
                
                # Reset error counter on successful update
                with QMutexLocker(self._state_mutex):
                    self._telemetry_errors = 0
                    
            else:
                # Handle failed telemetry reception
                with QMutexLocker(self._state_mutex):
                    self._telemetry_errors += 1
                    if self._telemetry_errors > 5:  # Alert after 5 consecutive failures
                        logging.warning("Multiple telemetry reception failures")
                        
        except CommunicationError as e:
            # Log error but don't show dialog every second
            logging.error(f"Failed to receive telemetry: {e}")
            with QMutexLocker(self._state_mutex):
                self._telemetry_errors += 1
        except Exception as e:
            logging.error(f"Unexpected error updating telemetry: {e}")
            with QMutexLocker(self._state_mutex):
                self._telemetry_errors += 1
    
    def init_backend_connection(self):
        """
        Thread-safe backend connection initialization with graceful fallback.
        """
        try:
            rocket_link.initialize_connection()
            mode = rocket_link.get_connection_mode()
            self.set_backend_connected(True, mode)
            self.set_system_status("Connected")
            logging.info(f"Backend connection established: {mode}")
        except CommunicationError as e:
            logging.warning(f"Failed to connect to backend: {e}")
            self.set_backend_connected(False, "Disconnected")
            self.set_system_status("Simulation Mode")
            QMessageBox.warning(
                self, 
                "Backend Connection", 
                "Failed to connect to backend. Running in simulation mode.\n\n"
                "You can still explore the GUI and see simulated telemetry data."
            )
    
    def update_connection_status(self):
        """
        Thread-safe status bar update with connection information.
        """
        try:
            connected = self.get_backend_connected()
            if connected:
                mode = rocket_link.get_connection_mode()
                if rocket_link.is_simulation_mode():
                    self.status_bar.showMessage(f"Connected: {mode} | Telemetry: Simulated", 0)
                    self.status_bar.setStyleSheet("background-color: #FFA500;")  # Orange for simulation
                else:
                    self.status_bar.showMessage(f"Connected: {mode} | Telemetry: Live", 0)
                    self.status_bar.setStyleSheet("background-color: #90EE90;")  # Light green for live
            else:
                self.status_bar.showMessage("Disconnected | No Telemetry", 0)
                self.status_bar.setStyleSheet("background-color: #FFB6C1;")  # Light red for disconnected
        except Exception as e:
            logging.error(f"Error updating connection status: {e}")
    
    def check_connection_status(self):
        """
        Periodic connection status check.
        """
        try:
            # Check if connection is still alive
            status = rocket_link.get_connection_mode()
            current_connected = self.get_backend_connected()
            
            # Update status if needed
            if status:
                if not current_connected:
                    self.set_backend_connected(True, status)
                    logging.info("Backend connection restored")
            else:
                if current_connected:
                    self.set_backend_connected(False, "Disconnected")
                    logging.warning("Backend connection lost")
                    
        except Exception as e:
            logging.error(f"Error checking connection status: {e}")
            # Assume disconnected on error
            if self.get_backend_connected():
                self.set_backend_connected(False, "Error")
    
    def closeEvent(self, event):
        """
        Thread-safe window close event handler with proper resource cleanup.
        """
        # Set shutdown flag
        with QMutexLocker(self._state_mutex):
            self._shutting_down = True
        
        # Stop timers
        if hasattr(self, 'telemetry_timer'):
            self.telemetry_timer.stop()
        if hasattr(self, 'render_timer'):
            self.render_timer.stop()
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        
        # Close backend connection
        if self.get_backend_connected():
            try:
                rocket_link.close_connection()
                self.set_backend_connected(False, "Shutdown")
            except Exception as e:
                logging.error(f"Error closing backend connection: {e}")
        
        # Close 3D view
        if hasattr(self, 'rocket_view'):
            try:
                self.rocket_view.close()
            except Exception as e:
                logging.error(f"Error closing rocket view: {e}")
        
        logging.info("MainWindow shutdown completed")
        event.accept()
    
    # Thread-safe signal handlers
    def _handle_telemetry_update(self, telemetry_data: Dict[str, Any]):
        """Handle telemetry update signal in main thread."""
        try:
            self.mission_control_panel.update_telemetry(telemetry_data)
        except Exception as e:
            logging.error(f"Error updating telemetry display: {e}")
    
    def _handle_connection_status_change(self, connected: bool, mode: str):
        """Handle connection status change signal in main thread."""
        try:
            self.update_connection_status()
        except Exception as e:
            logging.error(f"Error updating connection status display: {e}")
    
    def _handle_system_status_change(self, status: str):
        """Handle system status change signal in main thread."""
        try:
            self.mission_control_panel.update_status(status)
        except Exception as e:
            logging.error(f"Error updating system status display: {e}")
    
    def _handle_error_display(self, title: str, message: str):
        """Handle error display signal in main thread."""
        try:
            QMessageBox.warning(self, title, message)
        except Exception as e:
            logging.error(f"Error displaying error message: {e}")
    
    def _handle_rocket_position_update(self, position: tuple, orientation: tuple):
        """Handle rocket position update signal in main thread."""
        try:
            if hasattr(self, 'rocket_view') and self.rocket_view:
                data = {'position': position, 'orientation': orientation}
                self.rocket_view.update_telemetry(data)
        except Exception as e:
            logging.error(f"Error updating rocket position: {e}")