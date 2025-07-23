#!/usr/bin/env python3
"""
NovoGround Smooth Rocket Flight Demo

A polished rocket launch simulation that integrates seamlessly with the NovoGround
telemetry system. Features smooth data flow, realistic physics, and beautiful
visualization of a complete mission profile.
"""

import sys
import time
import threading
import math
import random
import numpy as np
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

# Import NovoGround components
from src.gui.main_window import MainWindow
from src.utils.logger import logger
from src.backend.cplusplus_bindings import backend


class SmoothRocketSimulation(QObject):
    """
    High-fidelity rocket simulation with smooth telemetry data generation.
    
    Mission Profile:
    - Pre-launch: 0-5s (system checks, final countdown)
    - Launch: 5-15s (motor burn, rapid acceleration)
    - Coast: 15-30s (ballistic trajectory to apogee)
    - Descent: 30-50s (free fall with increasing drag)
    - Recovery: 50-70s (parachute deployment and descent)
    - Landing: 70s+ (touchdown and post-flight)
    """
    
    # Signals for thread-safe communication
    telemetry_ready = pyqtSignal(dict)
    mission_event = pyqtSignal(str, str)  # phase, message
    
    def __init__(self):
        super().__init__()
        
        # Simulation parameters
        self.dt = 0.05  # 50ms timestep = 20Hz
        self.mission_time = 0.0
        self.running = False
        
        # Mission timeline (seconds)
        self.phase_times = {
            'pre_launch': (0, 5),
            'launch': (5, 15),
            'coast': (15, 30),
            'descent': (30, 50),
            'recovery': (50, 70),
            'landed': (70, float('inf'))
        }
        
        # Physics state
        self.reset_rocket_state()
        
        # Rocket parameters (realistic model rocket)
        self.dry_mass = 1.2  # kg
        self.prop_mass_initial = 0.8  # kg
        self.thrust_curve = self._generate_thrust_curve()
        self.drag_coeff = 0.75
        self.reference_area = 0.006  # m^2 (approx 3" diameter)
        self.recovery_area = 0.5  # m^2 (parachute)
        
        # System state
        self.current_phase = 'pre_launch'
        self.battery_level = 100.0
        self.max_altitude_reached = 0.0
        self.recovery_deployed = False
        
        # Smoothing filters for realistic sensor data
        self.velocity_filter = ExponentialFilter(alpha=0.7)
        self.accel_filter = ExponentialFilter(alpha=0.8)
        self.altitude_filter = ExponentialFilter(alpha=0.9)
        
        logger.log_event("🚀 Smooth rocket simulation initialized", "INFO")
    
    def reset_rocket_state(self):
        """Reset rocket to launch pad conditions."""
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z meters
        self.velocity = np.array([0.0, 0.0, 0.0])  # m/s
        self.acceleration = np.array([0.0, 0.0, -9.81])  # m/s^2
        self.orientation = np.array([0.0, 0.0, 0.0])  # pitch, yaw, roll degrees
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # deg/s
    
    def _generate_thrust_curve(self):
        """Generate realistic motor thrust curve."""
        # Typical Estes D12 motor curve (scaled)
        times = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0])
        thrusts = np.array([0, 25, 18, 15, 12, 8, 4, 2, 0, 0])  # Newtons
        return {'time': times, 'thrust': thrusts}
    
    def start_mission(self):
        """Begin the rocket flight simulation."""
        self.running = True
        self.mission_time = 0.0
        self.current_phase = 'pre_launch'
        self.reset_rocket_state()
        self.battery_level = 100.0
        self.max_altitude_reached = 0.0
        self.recovery_deployed = False
        
        # Start simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.timer.start(int(self.dt * 1000))  # Convert to milliseconds
        
        self.mission_event.emit('pre_launch', '🚀 T-5: Final systems check - All systems GO!')
        logger.log_event("Rocket launch sequence initiated", "INFO")
    
    def stop_mission(self):
        """Stop the simulation."""
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.mission_event.emit('aborted', '🛑 Mission aborted by operator')
        logger.log_event("Rocket mission stopped", "WARNING")
    
    def simulation_step(self):
        """Execute one simulation timestep."""
        if not self.running:
            return
        
        self.mission_time += self.dt
        
        # Update mission phase
        self._update_mission_phase()
        
        # Calculate physics
        self._update_physics()
        
        # Generate telemetry with realistic sensor characteristics
        telemetry = self._generate_smooth_telemetry()
        
        # Emit telemetry data
        self.telemetry_ready.emit(telemetry)
        
        # Check for mission completion
        if self.mission_time > 75 or (self.position[2] <= 0 and self.mission_time > 70):
            self._complete_mission()
    
    def _update_mission_phase(self):
        """Update current mission phase based on time and conditions."""
        old_phase = self.current_phase
        
        for phase, (start_time, end_time) in self.phase_times.items():
            if start_time <= self.mission_time < end_time:
                self.current_phase = phase
                break
        
        # Special conditions
        if self.position[2] <= 0 and self.mission_time > 70:
            self.current_phase = 'landed'
        
        # Emit phase change events
        if old_phase != self.current_phase:
            self._emit_phase_event(self.current_phase)
    
    def _emit_phase_event(self, phase):
        """Emit mission phase change events."""
        events = {
            'launch': f'🔥 T+{self.mission_time:.1f}s: LIFTOFF! Motor ignition detected',
            'coast': f'⬆️ T+{self.mission_time:.1f}s: Motor burnout - Coasting to apogee',
            'descent': f'📐 T+{self.mission_time:.1f}s: Apogee! Peak altitude: {self.max_altitude_reached:.0f}m',
            'recovery': f'🪂 T+{self.mission_time:.1f}s: Recovery system deployed',
            'landed': f'🛬 T+{self.mission_time:.1f}s: Landing detected - Mission complete!'
        }
        
        if phase in events:
            self.mission_event.emit(phase, events[phase])
    
    def _update_physics(self):
        """Calculate realistic rocket physics with smooth integration."""
        # Environmental constants
        g = 9.81  # m/s^2
        air_density = self._get_air_density(self.position[2])
        
        # Reset forces
        forces = np.array([0.0, 0.0, -self.get_current_mass() * g])  # Gravity
        
        # Phase-specific forces
        if self.current_phase == 'pre_launch':
            # Small vibrations on launch pad
            forces += np.random.normal(0, 0.1, 3)
            
        elif self.current_phase == 'launch':
            # Motor thrust
            thrust = self._get_motor_thrust()
            thrust_vector = np.array([0, 0, thrust])
            
            # Add small random variations (wind, thrust vector variation)
            thrust_vector[0] += random.uniform(-0.5, 0.5)
            thrust_vector[1] += random.uniform(-0.3, 0.3)
            forces += thrust_vector
            
            # Update orientation during boost
            self.orientation[0] += random.uniform(-0.2, 0.2)  # Pitch wobble
            self.orientation[2] += random.uniform(-0.1, 0.1)  # Roll
            
        else:
            # Drag force (coast, descent, recovery)
            velocity_mag = np.linalg.norm(self.velocity)
            if velocity_mag > 0:
                drag_coeff = self.drag_coeff
                drag_area = self.reference_area
                
                # Increase drag during recovery
                if self.current_phase == 'recovery':
                    drag_coeff = 1.2
                    drag_area = self.recovery_area
                    if not self.recovery_deployed:
                        self.recovery_deployed = True
                
                drag_force = 0.5 * air_density * drag_coeff * drag_area * velocity_mag**2
                drag_direction = -self.velocity / velocity_mag
                forces += drag_force * drag_direction
        
        # Integrate motion using Verlet integration for stability
        acceleration = forces / self.get_current_mass()
        
        # Smooth integration
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt + 0.5 * acceleration * self.dt**2
        
        # Ground constraint
        if self.position[2] <= 0:
            self.position[2] = 0
            if self.velocity[2] < 0:
                self.velocity[2] *= -0.3  # Bounce with energy loss
                if abs(self.velocity[2]) < 1.0:
                    self.velocity[2] = 0
        
        # Update derived quantities
        self.acceleration = acceleration
        self.max_altitude_reached = max(self.max_altitude_reached, self.position[2])
        
        # Battery discharge
        if self.current_phase != 'pre_launch':
            discharge_rate = 0.5 if self.current_phase == 'launch' else 0.1
            self.battery_level -= discharge_rate * self.dt
            self.battery_level = max(0, self.battery_level)
    
    def _get_air_density(self, altitude):
        """Calculate air density at given altitude."""
        # Simplified atmospheric model
        return 1.225 * math.exp(-altitude / 8400)  # kg/m^3
    
    def get_current_mass(self):
        """Calculate current rocket mass including propellant consumption."""
        if self.current_phase == 'launch':
            burn_fraction = (self.mission_time - 5) / 10  # 10s burn time
            remaining_prop = self.prop_mass_initial * max(0, 1 - burn_fraction)
            return self.dry_mass + remaining_prop
        return self.dry_mass
    
    def _get_motor_thrust(self):
        """Get motor thrust at current time."""
        burn_time = self.mission_time - 5  # Launch starts at T+5
        if burn_time < 0 or burn_time > 10:
            return 0
        
        # Interpolate thrust curve
        return np.interp(burn_time, self.thrust_curve['time'], self.thrust_curve['thrust'])
    
    def _generate_smooth_telemetry(self):
        """Generate smooth, realistic telemetry data."""
        # Apply filtering for smooth sensor readings
        filtered_velocity = self.velocity_filter.update(self.velocity)
        filtered_accel = self.accel_filter.update(self.acceleration)
        filtered_altitude = self.altitude_filter.update(self.position[2])
        
        # Add realistic sensor noise
        pos_noise = np.random.normal(0, 0.1, 3)  # GPS noise
        vel_noise = np.random.normal(0, 0.05, 3)  # IMU noise
        accel_noise = np.random.normal(0, 0.2, 3)  # Accelerometer noise
        
        # Calculate derived values
        speed = np.linalg.norm(filtered_velocity)
        accel_magnitude = np.linalg.norm(filtered_accel)
        
        # Status flags with realistic behavior
        status_flags = {
            "motor_failure": False,
            "sensor_error": random.random() < 0.001,  # Very rare
            "system_health": self.battery_level > 10,
            "recovery_deployed": self.recovery_deployed,
            "mission_active": self.current_phase not in ['pre_launch', 'landed'],
            "low_battery": self.battery_level < 20,
            "apogee_detected": self.current_phase == 'descent'
        }
        
        # Voltage based on battery level
        base_voltage = 12000 + random.uniform(-50, 50)  # millivolts
        voltage = int(base_voltage * (self.battery_level / 100))
        
        return {
            "timestamp": time.time(),
            "mission_time": round(self.mission_time, 2),
            "position": (self.position + pos_noise).tolist(),
            "velocity": (filtered_velocity + vel_noise).tolist(),
            "acceleration": (filtered_accel + accel_noise).tolist(),
            "orientation": self.orientation.tolist(),
            "altitude": float(filtered_altitude),
            "speed": float(speed),
            "acceleration_magnitude": float(accel_magnitude),
            "max_altitude": float(self.max_altitude_reached),
            "voltage": voltage,
            "battery_level": round(self.battery_level, 1),
            "mission_phase": self.current_phase,
            "motor_thrust": float(self._get_motor_thrust()),
            "mass": float(self.get_current_mass()),
            "status_flags": status_flags,
            "telemetry_rate": 20.0,  # 20 Hz
            "signal_strength": random.uniform(0.85, 1.0),
            "data_quality": 1.0 - (0.001 if status_flags["sensor_error"] else 0)
        }
    
    def _complete_mission(self):
        """Complete the mission and stop simulation."""
        self.running = False
        self.timer.stop()
        
        # Final mission statistics
        final_stats = {
            'mission_duration': self.mission_time,
            'max_altitude': self.max_altitude_reached,
            'max_speed': np.linalg.norm(self.velocity),
            'landing_velocity': np.linalg.norm(self.velocity) if self.position[2] <= 0 else 0
        }
        
        stats_msg = (f"🎯 Mission Complete!\n"
                    f"Duration: {final_stats['mission_duration']:.1f}s\n"
                    f"Max Altitude: {final_stats['max_altitude']:.1f}m\n"
                    f"Max Speed: {final_stats['max_speed']:.1f}m/s")
        
        self.mission_event.emit('complete', stats_msg)
        logger.log_event(f"Mission completed: {stats_msg.replace(chr(10), ' ')}", "INFO")


class ExponentialFilter:
    """Simple exponential filter for smooth sensor data."""
    
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_value = None
    
    def update(self, value):
        if self.prev_value is None:
            self.prev_value = value
            return value
        
        filtered = self.alpha * value + (1 - self.alpha) * self.prev_value
        self.prev_value = filtered
        return filtered


class RocketSimulationInterface:
    """Interface between simulation and NovoGround backend."""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.latest_telemetry = None
        
        # Connect simulation to interface
        simulation.telemetry_ready.connect(self.on_telemetry_received)
        simulation.mission_event.connect(self.on_mission_event)
    
    def on_telemetry_received(self, telemetry):
        """Receive telemetry from simulation."""
        self.latest_telemetry = telemetry
        
        # Inject into backend simulation system
        if hasattr(backend, 'simulation_data'):
            backend.simulation_data.update(telemetry)
    
    def on_mission_event(self, phase, message):
        """Handle mission events."""
        print(f"\n{message}")
        logger.log_event(message, "INFO")


def main():
    """Launch NovoGround with smooth rocket simulation."""
    print("🚀 NovoGround - Smooth Rocket Flight Demo")
    print("=" * 55)
    print("Features:")
    print("  • Realistic physics simulation (20Hz)")
    print("  • Smooth telemetry data flow")
    print("  • Complete mission profile")
    print("  • Beautiful data visualization")
    print("  • Thread-safe operation")
    print("=" * 55)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    # Wait for GUI initialization
    app.processEvents()
    time.sleep(1.0)
    
    # Create simulation
    simulation = SmoothRocketSimulation()
    interface = RocketSimulationInterface(simulation)
    
    # Auto-start mission after brief delay
    def auto_launch():
        time.sleep(2)
        print("\n🚀 Auto-launching in 3 seconds...")
        time.sleep(1)
        print("⚠️  3...")
        time.sleep(1)
        print("⚠️  2...")
        time.sleep(1)
        print("⚠️  1...")
        time.sleep(1)
        simulation.start_mission()
    
    # Start auto-launch in background thread
    launch_thread = threading.Thread(target=auto_launch, daemon=True)
    launch_thread.start()
    
    print("\n🎮 Controls:")
    print("  • Simulation will auto-start in 5 seconds")
    print("  • Use Mission Control panel to abort if needed")
    print("  • Watch telemetry dashboard for live data")
    print("  • View 3D rocket position in render window")
    print("\n📊 This demo shows ~75 seconds of flight data")
    print("   with smooth, realistic rocket physics.\n")
    
    # Run Qt event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        simulation.stop_mission()
        sys.exit(0)


if __name__ == "__main__":
    main()