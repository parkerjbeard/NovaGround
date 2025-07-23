from panda3d.core import WindowProperties, PerspectiveLens, Vec3, NodePath, AmbientLight, DirectionalLight
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from .shaders import ShaderProgram
from .grid_view import GridView
from direct.task import Task
import logging
import sys
import threading
import time
from typing import Dict, Any, Optional

class RocketGroup(NodePath):
    """
    Custom rendering group that manages shader program usage and model matrix
    application for each part of the rocket model.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.shader_program = None

    def set_shader(self, shader: ShaderProgram):
        """
        Sets the shader program for the group and all its children.
        """
        self.shader_program = shader
        super().set_shader(shader.get_shader())  # Correctly call superclass method
        for child in self.get_children():
            child.set_shader(shader.get_shader())

class RocketView(ShowBase):
    """
    Thread-safe 3D rocket model renderer. Handles window creation and management,
    loading and rendering of the 3D rocket model, camera and projection setup, user input
    for model manipulation, shader program management, lighting setup, grid rendering,
    and thread-safe model transformation updates based on telemetry data.
    """

    def __init__(self, parent_handle, model_path: str = None):
        # Ensure parent_handle is an integer
        if not isinstance(parent_handle, int):
            raise TypeError("parent_handle must be an integer.")
        
        # Thread safety infrastructure
        self._telemetry_lock = threading.RLock()
        self._rocket_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        
        # Thread-safe telemetry state
        self._current_telemetry = {}
        self._last_update_time = 0
        self._telemetry_timeout = 5.0  # seconds
        self._is_shutting_down = False
        
        # Initialize Panda3D without creating a new window
        loadPrcFileData("", "window-type none")
        super().__init__(windowType="none")
        logging.info("Panda3D ShowBase initialized with windowType='none'")

        # Open a new window embedded in the Qt widget
        props = WindowProperties()
        props.set_parent_window(parent_handle)  # Now correctly receives an int
        props.set_title("3D Rocket Visualization")
        props.set_size(800, 600)  # Adjust size as needed
        try:
            self.open_window(props=props)
            logging.info("Panda3D window opened successfully with parent_handle.")
        except Exception as e:
            logging.exception("Failed to open Panda3D window.")
            raise
        
        # Disable default camera controls
        self.disable_mouse()
        logging.info("Default camera controls disabled.")

        # Load shaders
        self.basic_shader = ShaderProgram(vertex_shader_file="src/shaders/basic.vert", fragment_shader_file="src/shaders/basic.frag")
        logging.info("Basic shader loaded.")

        # Setup lighting
        self._setup_lighting()
        logging.info("Lighting setup completed.")

        # Setup camera
        self._setup_camera()
        logging.info("Camera setup completed.")

        # Load grid (now a 3D environment)
        self.grid = GridView(self.render)
        logging.info("GridView (3D Environment) initialized and added to render")

        # Load rocket model if model_path is provided
        if model_path:
            self.rocket = self._load_model(model_path)
            self.rocket.reparent_to(self.render)
            logging.info("Rocket model loaded and reparented to render.")

            # Apply shader to rocket
            self.rocket.set_shader(self.basic_shader.get_shader())
            logging.info("Shader applied to rocket model.")
        else:
            self.rocket = None
            logging.info("No rocket model file provided. Skipping rocket loading.")

        # User input keys
        self.keys = {
            "rotate_left": "a",
            "rotate_right": "d",
            "rotate_up": "w",
            "rotate_down": "s",
            "translate_forward": "arrow_up",
            "translate_backward": "arrow_down",
            "translate_left": "arrow_left",
            "translate_right": "arrow_right"
        }
        self._setup_controls()
        logging.info("User controls setup completed.")

        # Task to update model based on telemetry
        self.taskMgr.add(self.update_telemetry_task, "UpdateTelemetryTask")
        logging.info("Telemetry update task added.")

        # Add a task to continuously update the scene
        self.taskMgr.add(self.update_scene, "UpdateSceneTask")

    def _setup_lighting(self):
        """
        Sets up ambient and directional lighting in the scene.
        """
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color((0.5, 0.5, 0.5, 1))
        ambient_light_np = self.render.attach_new_node(ambient_light)
        self.render.set_light(ambient_light_np)
        logging.info("Ambient light configured.")

        directional_light = DirectionalLight("directional_light")
        directional_light.set_direction(Vec3(-1, -1, -2))
        directional_light.set_color((0.7, 0.7, 0.7, 1))
        directional_light_np = self.render.attach_new_node(directional_light)
        self.render.set_light(directional_light_np)
        logging.info("Directional light configured.")

    def _setup_camera(self):
        """
        Configures the camera's position and perspective.
        """
        self.camera.set_pos(0, -100, 0)  # Move the camera back and level with the ground
        self.camera.look_at(0, 0, 0)
        lens = PerspectiveLens()
        lens.set_fov(60)
        self.cam.node().set_lens(lens)
        logging.info(f"Camera positioned at {self.camera.get_pos()} looking at (0, 0, 0)")

    def _load_model(self, model_path: str) -> NodePath:
        """
        Loads a 3D model from the given path.
        """
        model = self.loader.loadModel(model_path)
        if not model:
            self.exit_with_error(f"Failed to load model: {model_path}")
        logging.info(f"Model loaded from {model_path}")
        return model

    def _setup_controls(self):
        """
        Sets up keyboard controls for interacting with the rocket model.
        """
        for action, key in self.keys.items():
            self.accept(key, self.on_key_press, [action])
        logging.info("Keyboard controls accepted.")

    def on_key_press(self, action: str):
        """
        Thread-safe key press event handler for model manipulation.
        """
        try:
            with self._rocket_lock:
                if not self.rocket:
                    logging.warning("Rocket model is not loaded. Ignoring input.")
                    return

                if action == "rotate_left":
                    self.rocket.set_h(self.rocket.get_h() + 5)
                elif action == "rotate_right":
                    self.rocket.set_h(self.rocket.get_h() - 5)
                elif action == "rotate_up":
                    self.rocket.set_p(self.rocket.get_p() + 5)
                elif action == "rotate_down":
                    self.rocket.set_p(self.rocket.get_p() - 5)
                elif action == "translate_forward":
                    self.rocket.set_y(self.rocket, 1)
                elif action == "translate_backward":
                    self.rocket.set_y(self.rocket, -1)
                elif action == "translate_left":
                    self.rocket.set_x(self.rocket, -1)
                elif action == "translate_right":
                    self.rocket.set_x(self.rocket, 1)
                logging.info(f"Action performed: {action}")
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def update_telemetry_task(self, task: Task) -> Task:
        """
        Periodic task to update the rocket model based on telemetry data.
        Runs in Panda3D's task manager.
        """
        try:
            # Check if shutting down
            with self._shutdown_lock:
                if self._is_shutting_down:
                    return Task.done
            
            # Apply any pending telemetry updates
            self._apply_telemetry_updates()
            
        except Exception as e:
            logging.error(f"Error in telemetry update task: {e}")
        
        return Task.cont
    
    def _apply_telemetry_updates(self):
        """
        Apply telemetry updates to the rocket model (thread-safe).
        """
        with self._telemetry_lock:
            # Check if we have recent telemetry data
            current_time = time.time()
            if not self._current_telemetry:
                return
            
            # Check if data is too old
            if current_time - self._last_update_time > self._telemetry_timeout:
                logging.warning("Telemetry data is stale")
                return
            
            # Apply updates to rocket model
            with self._rocket_lock:
                if self.rocket:
                    try:
                        position = self._current_telemetry.get('position')
                        orientation = self._current_telemetry.get('orientation')
                        
                        if position:
                            # Convert to Panda3D coordinates if needed
                            if isinstance(position, (list, tuple)) and len(position) >= 3:
                                self.rocket.set_pos(Vec3(position[0], position[1], position[2]))
                        
                        if orientation:
                            # Convert to Panda3D coordinates if needed
                            if isinstance(orientation, (list, tuple)) and len(orientation) >= 3:
                                self.rocket.set_hpr(Vec3(orientation[0], orientation[1], orientation[2]))
                                
                    except Exception as e:
                        logging.error(f"Error applying telemetry updates: {e}")
    
    def update_telemetry(self, telemetry_data: Dict[str, Any]):
        """
        Thread-safe telemetry update method called from main thread.
        
        :param telemetry_data: New telemetry data to apply
        """
        try:
            with self._telemetry_lock:
                # Update current telemetry data
                self._current_telemetry.update(telemetry_data)
                self._last_update_time = time.time()
                
                logging.debug(f"Telemetry updated: {telemetry_data}")
                
        except Exception as e:
            logging.error(f"Error updating telemetry: {e}")
    
    def get_current_telemetry(self) -> Dict[str, Any]:
        """
        Thread-safe getter for current telemetry data.
        
        :return: Copy of current telemetry data
        """
        with self._telemetry_lock:
            return self._current_telemetry.copy()
    
    def is_telemetry_fresh(self) -> bool:
        """
        Check if current telemetry data is fresh.
        
        :return: True if telemetry is recent, False otherwise
        """
        with self._telemetry_lock:
            if not self._current_telemetry:
                return False
            return time.time() - self._last_update_time <= self._telemetry_timeout

    def get_rocket_state(self) -> Dict[str, Any]:
        """
        Thread-safe getter for current rocket model state.
        
        :return: Current rocket position and orientation
        """
        with self._rocket_lock:
            if self.rocket:
                return {
                    'position': tuple(self.rocket.get_pos()),
                    'orientation': tuple(self.rocket.get_hpr())
                }
            else:
                return {
                    'position': (0, 0, 0),
                    'orientation': (0, 0, 0)
                }

    def exit_with_error(self, message: str):
        """
        Exits the application with an error message.
        """
        logging.error(f"Error: {message}")
        print(f"Error: {message}")
        sys.exit(1)

    def close(self):
        """
        Thread-safe close method to ensure Panda3D shuts down gracefully.
        """
        logging.info("Closing RocketView and shutting down Panda3D.")
        
        # Set shutdown flag
        with self._shutdown_lock:
            self._is_shutting_down = True
        
        try:
            # Stop all tasks
            self.taskMgr.remove("UpdateTelemetryTask")
            self.taskMgr.remove("UpdateSceneTask")
        except Exception as e:
            logging.warning(f"Error stopping tasks: {e}")
        
        try:
            # Clean shutdown
            self.userExit()
            super().close()
        except Exception as e:
            logging.error(f"Error during RocketView shutdown: {e}")

    def update_scene(self, task):
        """
        Scene update task that runs continuously.
        """
        try:
            # Check if shutting down
            with self._shutdown_lock:
                if self._is_shutting_down:
                    return Task.done
            
            # Scene update logic can go here
            # Currently just continuing the task
            
        except Exception as e:
            logging.error(f"Error in scene update task: {e}")
        
        return Task.cont