#!/usr/bin/env python3
"""
NovoGround Rocket Launch Simulation

This script launches the NovoGround GUI with a realistic rocket flight simulation
that generates telemetry data for a complete mission profile:
- Pre-launch (5 seconds)
- Launch and boost phase (10 seconds) 
- Coast phase (15 seconds)
- Apogee and descent (20 seconds)
- Recovery deployment (10 seconds)
- Landing (remaining time)

The simulation includes realistic physics, sensor noise, and mission events.
"""

import sys
import time
import threading
import math
import random
import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import NovoGround components
from src.gui.main_window import MainWindow
from src.utils.logger import logger


class RocketFlightSimulation:
    """
    Realistic rocket flight simulation with complete mission profile.
    """
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.mission_time = 0.0
        self.simulation_running = False
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        
        # Flight parameters
        self.launch_time = 5.0  # Pre-launch time
        self.burn_time = 10.0   # Motor burn duration
        self.coast_time = 15.0  # Coast to apogee
        self.descent_time = 20.0  # Descent phase
        self.recovery_deploy_time = 45.0  # Recovery deployment
        self.landing_time = 65.0  # Total mission time
        
        # Physics state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.velocity = np.array([0.0, 0.0, 0.0])  # m/s
        self.acceleration = np.array([0.0, 0.0, 0.0])  # m/s^2
        self.orientation = np.array([0.0, 0.0, 0.0])  # pitch, yaw, roll in degrees
        
        # Rocket parameters
        self.dry_mass = 5.0  # kg
        self.propellant_mass = 2.0  # kg
        self.thrust = 500.0  # Newtons
        self.drag_coefficient = 0.5
        self.reference_area = 0.01  # m^2
        
        # Mission state
        self.mission_phase = "pre_launch"
        self.max_altitude = 0.0
        self.battery_voltage = 12500  # millivolts
        self.recovery_deployed = False
        
        logger.log_event("Rocket flight simulation initialized", "INFO")
    
    def start_simulation(self):
        """Start the rocket flight simulation."""
        self.simulation_running = True
        self.mission_time = 0.0
        self.mission_phase = "pre_launch"
        self.simulation_timer.start(100)  # 10 Hz update rate
        logger.log_event("🚀 Rocket launch simulation started!", "INFO")
        print("\n" + "="*60)
        print("🚀 ROCKET LAUNCH SIMULATION STARTED!")
        print("="*60)
        print("Mission Profile:")
        print(f"  • Pre-launch:     0-{self.launch_time}s")
        print(f"  • Launch/Boost:   {self.launch_time}-{self.launch_time + self.burn_time}s")
        print(f"  • Coast:          {self.launch_time + self.burn_time}-{self.launch_time + self.burn_time + self.coast_time}s")
        print(f"  • Descent:        {self.launch_time + self.burn_time + self.coast_time}-{self.recovery_deploy_time}s")
        print(f"  • Recovery:       {self.recovery_deploy_time}-{self.landing_time}s")
        print("="*60)
    
    def stop_simulation(self):
        """Stop the rocket flight simulation."""
        self.simulation_running = False
        self.simulation_timer.stop()
        logger.log_event("Rocket launch simulation stopped", "INFO")
        print("🛬 Simulation completed - Rocket landed safely!")
    
    def update_simulation(self):
        """Update simulation physics and telemetry."""
        if not self.simulation_running:
            return
        
        dt = 0.1  # 100ms time step
        self.mission_time += dt
        
        # Update mission phase based on time
        self._update_mission_phase()
        
        # Calculate physics based on current phase
        self._calculate_physics(dt)
        
        # Add realistic sensor noise
        self._add_sensor_noise()
        
        # Generate and send telemetry
        telemetry_data = self._generate_telemetry()
        self._send_telemetry(telemetry_data)
        
        # Print mission updates
        self._print_mission_status()
        
        # Check for mission end
        if self.mission_time >= self.landing_time:
            self.stop_simulation()
    
    def _update_mission_phase(self):
        """Update the current mission phase based on time."""
        if self.mission_time < self.launch_time:
            self.mission_phase = "pre_launch"
        elif self.mission_time < self.launch_time + self.burn_time:
            if self.mission_phase != "boost":
                print(f"🔥 T+{self.mission_time:.1f}s: LIFTOFF! Motor ignition")
            self.mission_phase = "boost"
        elif self.mission_time < self.launch_time + self.burn_time + self.coast_time:
            if self.mission_phase != "coast":
                print(f"⬆️ T+{self.mission_time:.1f}s: Motor burnout - Coasting to apogee")
            self.mission_phase = "coast"
        elif self.mission_time < self.recovery_deploy_time:
            if self.mission_phase != "descent":
                print(f"⬇️ T+{self.mission_time:.1f}s: Apogee reached - Beginning descent")
            self.mission_phase = "descent"
        elif self.mission_time < self.landing_time:
            if self.mission_phase != "recovery":
                print(f"🪂 T+{self.mission_time:.1f}s: Recovery system deployed!")
                self.recovery_deployed = True
            self.mission_phase = "recovery"
        else:
            self.mission_phase = "landed"
    
    def _calculate_physics(self, dt):
        """Calculate realistic rocket physics."""
        g = 9.81  # gravity
        air_density = 1.225  # kg/m^3 at sea level
        
        # Reset acceleration
        self.acceleration = np.array([0.0, 0.0, -g])  # Start with gravity
        
        if self.mission_phase == "pre_launch":
            # Rocket on pad - small vibrations
            self.acceleration += np.random.normal(0, 0.1, 3)
            
        elif self.mission_phase == "boost":
            # Motor thrust (vertical with small random variations)
            current_mass = self.dry_mass + self.propellant_mass * (1 - (self.mission_time - self.launch_time) / self.burn_time)
            thrust_acceleration = self.thrust / current_mass
            
            # Add thrust in Z direction with small variations
            self.acceleration[2] += thrust_acceleration + random.uniform(-20, 20)
            self.acceleration[0] += random.uniform(-2, 2)  # Small wind effects
            self.acceleration[1] += random.uniform(-1, 1)
            
        elif self.mission_phase == "coast":
            # Only gravity and drag
            speed = np.linalg.norm(self.velocity)
            if speed > 0:
                drag_force = 0.5 * air_density * self.drag_coefficient * self.reference_area * speed**2
                drag_acceleration = drag_force / self.dry_mass
                drag_direction = -self.velocity / speed
                self.acceleration += drag_acceleration * drag_direction
            
        elif self.mission_phase == "descent":
            # Gravity and drag (higher drag during descent)
            speed = np.linalg.norm(self.velocity)
            if speed > 0:
                drag_force = 0.5 * air_density * self.drag_coefficient * self.reference_area * speed**2
                drag_acceleration = drag_force / self.dry_mass
                drag_direction = -self.velocity / speed
                self.acceleration += drag_acceleration * drag_direction
            
        elif self.mission_phase == "recovery":
            # Recovery system deployed - high drag
            speed = np.linalg.norm(self.velocity)
            if speed > 0:
                # Much higher drag with parachute
                recovery_drag_coeff = 1.5
                drag_force = 0.5 * air_density * recovery_drag_coeff * 1.0 * speed**2  # Larger area
                drag_acceleration = drag_force / self.dry_mass
                drag_direction = -self.velocity / speed
                self.acceleration += drag_acceleration * drag_direction
        
        # Integrate motion (but don't go below ground)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Ground constraint
        if self.position[2] <= 0:
            self.position[2] = 0
            if self.velocity[2] < 0:
                self.velocity[2] = 0
                if self.mission_phase != "landed":
                    self.mission_phase = "landed"
        
        # Update max altitude
        self.max_altitude = max(self.max_altitude, self.position[2])
        
        # Update orientation (simple model)
        if self.mission_phase == "boost":
            self.orientation[0] += random.uniform(-1, 1)  # Small pitch variations
            self.orientation[2] += random.uniform(-0.5, 0.5)  # Roll
        
        # Battery discharge
        if self.mission_phase != "pre_launch":
            self.battery_voltage -= random.uniform(0, 2)  # Gradual discharge
    
    def _add_sensor_noise(self):
        """Add realistic sensor noise to measurements."""
        # Position noise (GPS)
        position_noise = np.random.normal(0, 0.5, 3)
        self.position += position_noise
        
        # Velocity noise (IMU integration errors)
        velocity_noise = np.random.normal(0, 0.1, 3)
        self.velocity += velocity_noise
        
        # Acceleration noise (accelerometer)
        accel_noise = np.random.normal(0, 0.5, 3)
        self.acceleration += accel_noise
        
        # Orientation noise (gyroscope drift)
        orientation_noise = np.random.normal(0, 0.2, 3)
        self.orientation += orientation_noise
    
    def _generate_telemetry(self):
        """Generate telemetry data packet."""
        # Calculate derived values
        speed = np.linalg.norm(self.velocity)
        acceleration_magnitude = np.linalg.norm(self.acceleration)
        
        # Status flags based on mission state
        status_flags = {
            "motor_failure": False,
            "sensor_error": random.random() < 0.01,  # 1% chance of sensor error
            "system_health": self.battery_voltage > 10000,
            "recovery_deployed": self.recovery_deployed,
            "mission_active": self.mission_phase not in ["pre_launch", "landed"]
        }
        
        # Add occasional anomalies for testing
        if random.random() < 0.05:  # 5% chance
            if random.random() < 0.5:
                status_flags["sensor_error"] = True
            else:
                # Simulate brief communication dropout
                return None
        
        return {
            "timestamp": time.time(),
            "mission_time": self.mission_time,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "orientation": self.orientation.tolist(),
            "altitude": float(self.position[2]),
            "speed": float(speed),
            "acceleration_magnitude": float(acceleration_magnitude),
            "max_altitude": float(self.max_altitude),
            "voltage": int(self.battery_voltage),
            "mission_phase": self.mission_phase,
            "status_flags": status_flags,
            "telemetry_rate": 10.0,  # 10 Hz
            "signal_strength": random.uniform(0.7, 1.0)
        }
    
    def _send_telemetry(self, telemetry_data):
        """Send telemetry data to the main window."""
        if telemetry_data and hasattr(self.main_window, 'update_telemetry'):
            self.main_window.update_telemetry(telemetry_data)
    
    def _print_mission_status(self):
        """Print periodic mission status updates."""
        if int(self.mission_time * 10) % 50 == 0:  # Every 5 seconds
            altitude_ft = self.position[2] * 3.28084
            speed_mph = np.linalg.norm(self.velocity) * 2.23694
            print(f"T+{self.mission_time:5.1f}s | {self.mission_phase:>10} | "
                  f"Alt: {altitude_ft:6.0f}ft | Speed: {speed_mph:5.1f}mph | "
                  f"Batt: {self.battery_voltage:4.0f}mV")


def main():
    """Launch NovoGround with rocket simulation."""
    print("🚀 NovoGround Rocket Launch Simulation")
    print("=" * 50)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create main window
    main_window = MainWindow()
    main_window.show()
    
    # Wait for GUI to initialize
    app.processEvents()
    time.sleep(1.0)
    
    # Create and start simulation
    simulation = RocketFlightSimulation(main_window)
    
    # Add simulation controls to main window
    def start_launch():
        simulation.start_simulation()
    
    def abort_launch():
        simulation.stop_simulation()
    
    # Connect simulation to mission control buttons
    if hasattr(main_window, 'mission_control_panel'):
        main_window.mission_control_panel.launch_mission.connect(start_launch)
        main_window.mission_control_panel.abort_mission.connect(abort_launch)
    
    print("\n🎮 Controls:")
    print("  • Click 'Launch Mission' button to start simulation")
    print("  • Click 'Abort Mission' button to stop simulation")
    print("  • Watch the telemetry dashboard for live data")
    print("  • View the 3D rocket position in the render window")
    print("\n💡 The simulation will run automatically for ~65 seconds")
    print("   showing a complete rocket flight profile with realistic data.\n")
    
    # Auto-start simulation after 3 seconds
    def auto_start():
        print("🚀 Auto-starting simulation in 3 seconds...")
        time.sleep(3)
        simulation.start_simulation()
    
    # Start auto-launch in background thread
    auto_thread = threading.Thread(target=auto_start, daemon=True)
    auto_thread.start()
    
    # Start Qt event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        simulation.stop_simulation()
        sys.exit(0)


if __name__ == "__main__":
    main()