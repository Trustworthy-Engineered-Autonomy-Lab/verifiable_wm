"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        #self.force_mag = 30.0
        self.force_mag = 10.0

        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.screen = None
        self.screen_width = 600 
        self.screen_height = 400

        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
    # the default setting for reset
    # def reset(self):
    #     self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    #     self.steps_beyond_done = None
    #     return np.array(self.state)
    def reset(self):
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self):
        # if self.render_mode is None:
        #     gym.logger.warn(
        #         "You are calling render method without specifying any render mode. "
        #         "You can specify the render_mode at initialization, "
        #         f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
        #     )
        #     return
    
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
    
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
    
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
    
        if self.state is None:
            return None
    
        x = self.state
    
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
    
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
    
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
    
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
    
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
    
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
    
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
    
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            import pygame
    
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    # Default render and close function
    # def render(self, mode='human'):
    #     screen_width = 600
    #     screen_height = 400

    #     world_width = self.x_threshold * 2
    #     scale = screen_width /world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * 1.0
    #     cartwidth = 50.0
    #     cartheight = 30.0

    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         axleoffset = cartheight / 4.0
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
    #         pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole.set_color(.8, .6, .4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth / 2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(.5, .5, .8)
    #         self.viewer.add_geom(self.axle)
    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)

    #     if self.state is None:
    #         return None

    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[2])

    #     return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()


# ============================================================
# CARLA AEBS (brake system) environment.
# Ported from aebs_carla/carla_aebs_gym_env.py; the CARLA client
# connects to CARLA_HOST:CARLA_PORT (default localhost:2000, e.g.
# a Docker server). Setting CARLA_SERVER to a CarlaUE4.sh path
# instead launches a local server binary.
# ============================================================

import os
import queue
import subprocess
import time

from gym import ObservationWrapper
from gym.error import DependencyNotInstalled
from gym.spaces import Box

try:
    import carla
    from carla import Transform, Location, Rotation
except ImportError:  # CARLA client not installed; CarlaAEBSEnv unavailable
    carla = None


class ServerManagerBinary:
    """Launch/stop a local CarlaUE4 server binary (optional)."""

    def __init__(self, opt_dict):
        self._proc = None
        self._carla_server_binary = opt_dict['CARLA_SERVER']

    def reset(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc.communicate()
        command = "{} -RenderOffScreen >/dev/null".format(self._carla_server_binary)
        print("Starting CARLA server:", command)
        self._proc = subprocess.Popen(command, shell=True)

    def wait_until_ready(self, wait=10.0):
        time.sleep(wait)

    def stop(self):
        import psutil
        parent = psutil.Process(self._proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        self._proc.communicate()


class CarlaAEBSEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"],
                "render_fps": 30}

    def __init__(self, disable_rendering=True, image_observation=False, use_scenic=False):
        super().__init__()
        
        # gym setting
        if disable_rendering and image_observation:
            disable_rendering = False
            Warning("Image observation is enabled, switching rendering on.")
        self.disable_rendering = disable_rendering
        self.image_observation = image_observation

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if not image_observation:
            self.observation_space = spaces.Box(low = np.array([0.0, 0.0], dtype=np.float32),
                                                high=np.array([60.0, 30.0], dtype=np.float32), dtype=np.float32)

        self.screen_width = 800
        self.screen_height = 600
        if image_observation:
            self.observation_space = spaces.Dict({
                'velocity': spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32),
                'distance': spaces.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float32),
            })
            self.observation_space['image'] = spaces.Box(low=0,
                                                         high=255,
                                                         shape=(self.screen_height, self.screen_width, 3),
                                                         dtype=np.uint8)

        # carla setting

        ## carla server: launch our own binary if CARLA_SERVER is set,
        ## otherwise connect to an already-running server (e.g. Docker)
        self.carla_server = None
        if os.environ.get("CARLA_SERVER"):
            self.carla_server = ServerManagerBinary(
                {'CARLA_SERVER': os.environ["CARLA_SERVER"]}
            )
            self.carla_server.reset()
            self.carla_server.wait_until_ready(wait=30.0)

        ## host and port via CARLA_HOST / CARLA_PORT; the CARLA server
        ## runs on the decaf machine (Docker), matching the original setup
        self.client = carla.Client(
            os.environ.get("CARLA_HOST", "ece-d4100-w02.ad.ufl.edu"),
            int(os.environ.get("CARLA_PORT", "2000")),
        )
        self.client.set_timeout(60.0)
        print("Client connected to server")

        ## load town 01
        self.world = self.client.load_world('Town01')

        ## set synchronous mode
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        settings.substepping = True
        self.world.apply_settings(settings)

        self.world.set_weather(carla.WeatherParameters.CloudyNoon)

        self.bp_library = self.world.get_blueprint_library()
        self.actors = []
        self.action_sleep = 0.001
        self.screen = None
        self.clock = None
        self.use_scenic = use_scenic
        if self.use_scenic:
            import scenic
            import scenic.simulators.carla.utils.utils as utils
            self.scenario_file = "./utils/scenarios.scenic"

        # this should be measured before the simulation starts
        # fixed for ego vehicle -- cybertruck, and the obstacle -- audi.a2
        self.extending_offset = 4.989461421966553

    def compute_reward(self, distance, collision):
        if collision is not None:
            reward = -200.0 - math.sqrt(collision.x**2 + collision.y**2 + collision.z**2)/100.0
            info = {'is_collision': True, 'is_stop': True}
            print(f"Collision! Reward: {reward}")
        else:
            too_far_reward = -((distance-3.0)/120.0*400+30) * (distance>3.0) 
            too_close_reward = -(40.0)*(distance<1.0)
            reward = too_far_reward + too_close_reward
            info = {'is_collision': False, 'is_stop': True}
            print(f"Distance: {distance}, Reward: {reward}")
        
        return info, reward

    def generate_vehicles_position(self, distance):
        if not self.use_scenic:
            obstacle_pos = np.random.uniform(distance+6, 320.0).item() # +1 in order to prevent the collision error due to large vehicle
            obstacle_spawn_point = Transform(Location(x=392.1, y=obstacle_pos, z=0.00),
                                             Rotation(pitch=0.0, yaw=90.0, roll=0.0))
            ego_pos = obstacle_pos - distance.item() - self.extending_offset
            ego_spawn_point = Transform(Location(x=392.1, y=ego_pos, z=0.04),
                                        Rotation(pitch=0.0, yaw=90.0, roll=0.0))
        
        else:
            line = 'distance = {}'.format(-(distance.item() + self.extending_offset))
            with open(self.scenario_file, 'r') as f:
                lines = f.readlines()
            with open(self.scenario_file, 'w') as f:
                lines[0] = line + '\n'
                f.writelines(lines)

            scenario = scenic.scenarioFromFile(self.scenario_file)
            scene, _ = scenario.generate()

            ego_object = scene.objects[0]
            loc = utils.scenicToCarlaLocation(ego_object.position, world=self.world, blueprint=ego_object.blueprint)
            loc = carla.Location(x=loc.x, y=loc.y, z=0.04)
            rot = utils.scenicToCarlaRotation(ego_object.heading)
            ego_spawn_point = Transform(loc, rot)

            obstacle_object = scene.objects[1]
            loc = utils.scenicToCarlaLocation(obstacle_object.position, world=self.world, blueprint=obstacle_object.blueprint)
            loc = carla.Location(x=loc.x, y=loc.y, z=0.00)
            rot = utils.scenicToCarlaRotation(obstacle_object.heading)
            obstacle_spawn_point = Transform(loc, rot)
            
        return ego_spawn_point, obstacle_spawn_point
    
    def spawn_ego_vehicle(self, spawn_point):
        ego_bp = self.bp_library.filter('vehicle.tesla.cybertruck')[0]
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
        #yaw = spawn_point.rotation.yaw
        #v_x = velocity.item() * math.cos(yaw)
        #v_y = velocity.item() * math.sin(yaw)
        #self.ego_vehicle.set_target_velocity(carla.Vector3D(x=v_x, y=v_y, z=0))
        self.actors.append(self.ego_vehicle)
    
    def set_ego_vehicle_velocity(self, velocity):
        heading_direction = self.ego_vehicle.get_transform().get_forward_vector()
        #yaw = self.obstacle_vehicle.get_transform().rotation.yaw
        #v_x = velocity.item() * math.cos(heading_direction)
        #v_y = velocity.item() * math.sin(heading_direction)
        velocity = velocity.item() * heading_direction
        self.ego_vehicle.set_target_velocity(velocity)

        #self.ego_vehicle.set_target_velocity(carla.Vector3D(x=v_x, y=v_y, z=0))

    def spawn_obstacle_vehicle(self, spawn_point):
        obstacle_bp = self.bp_library.filter('vehicle.audi.a2')[0]
        self.obstacle_vehicle = self.world.try_spawn_actor(obstacle_bp, spawn_point)
        self.actors.append(self.obstacle_vehicle)

    def setup_sensors(self):
        self._queues = []

        # Remove the previous on_tick callback before registering a new one.
        # world.on_tick() accumulates callbacks permanently unless explicitly
        # removed — forgetting this causes one extra callback per reset(), so
        # every tick gets linearly slower with the number of resets.
        if hasattr(self, '_on_tick_id') and self._on_tick_id is not None:
            try:
                self.world.remove_on_tick(self._on_tick_id)
            except Exception:
                pass
            self._on_tick_id = None

        def make_queue(register_event):
            q = queue.Queue()
            callback_id = register_event(q.put)
            self._queues.append(q)
            return callback_id

        self._on_tick_id = make_queue(self.world.on_tick)
        
        # the camera will be disabled when it is not needed
        if not self.disable_rendering:
            camera_bp = self.bp_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '100')
            self.camera = self.world.try_spawn_actor(
                            camera_bp,
                            #Transform(Location(x=0.8,y=0.0,z=1.7), Rotation(yaw=0.0)), 
                            Transform(Location(x=1.8,y=0.0,z=2.2), Rotation(yaw=0.0)), 
                            attach_to=self.ego_vehicle)
            self.actors.append(self.camera)
            make_queue(self.camera.listen)
        

        collision_bp = self.bp_library.find('sensor.other.collision')
        self.collision = self.world.try_spawn_actor(
                        collision_bp,
                        Transform(),
                        attach_to=self.ego_vehicle)
        self.actors.append(self.collision)
        self.collision_queue = queue.Queue()
        self.collision.listen(self.collision_queue.put)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def tick(self, timeout):
        time.sleep(self.action_sleep)
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def step(self, action):
        # call carla api to run one step
        #action_diff = abs(action.item() - self.action_state)
        self.action_state = action.item()
        #print(self.action_state)
        #self.action_state = 0.2*action.item() + self.action_state
        self.action_state = np.clip(self.action_state, 0.0, 1.0)
        control = carla.VehicleControl(throttle=0.0, 
                                       steer=0.0,
                                       brake=self.action_state)
        self.ego_vehicle.apply_control(control)
        
        # retrieve data from sensors 
        if not self.disable_rendering:
            snapshot, image = self.tick(timeout=2.0)
        else:
            snapshot = self.tick(timeout=2.0)[0]

        if not self.collision_queue.empty():
            collision = self.collision_queue.get().normal_impulse
            is_collision = True
        else:
            collision = None
            is_collision = False
        
        distance = self._compute_distance()
        velocity = self._compute_velocity()

        if self.image_observation:
            self.image = self._get_image_raw_data(image)
            #image.save_to_disk("{}.png".format(str(self.step_num).zfill(3)))
            self.step_num += 1 
            obs = {'velocity': velocity, 'distance': distance}
            obs['image'] = self.image
        else:
            obs = np.array([distance, velocity], dtype=np.float32)

        is_stop = velocity <= 0.0
        done = is_collision or is_stop

        # compute reward
        if done:
            info, reward = self.compute_reward(distance, collision)
        else:
            reward = 0.0
            info = {'is_collision': False, 'is_stop': False}
        
        # penalize the agent for making instant brake
        #if action_diff > 0.2:
        #    reward -= 5.0
        
        return obs, reward, done, info

    def _get_image_raw_data(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        return array

    def _compute_distance(self):
        distance = math.hypot(self.ego_vehicle.get_location().x-self.obstacle_vehicle.get_location().x,
                      self.ego_vehicle.get_location().y-self.obstacle_vehicle.get_location().y) \
                      - self.extending_offset
                      #- self.ego_vehicle.bounding_box.extent.x - self.leading_vehicle.bounding_box.extent.x
        #distance = self.obstacle_vehicle.get_location().y - self.ego_vehicle.get_location().y \
        #    - self.extending_offset
        #    #- self.ego_vehicle.bounding_box.extent.x - self.obstacle_vehicle.bounding_box.extent.x
        return distance
    
    def _compute_velocity(self):
        velocity = math.hypot(self.ego_vehicle.get_velocity().x, self.ego_vehicle.get_velocity().y)
        return velocity

    # def reset(self):
    #     # Clean all vehicles before reset the environment
    #     for i, _ in enumerate(self.actors):
    #         if self.actors[i] is not None:
    #             self.actors[i].destroy()
    #             self.actors[i] = None
    #     self.actors = []
    #     self.step_num = 0


    #     init_distance_bound = [45.0, 60.0]
    #     init_velocity_bound = [10.0, 25.0]
        
    #     while True:
    #         distance = np.array([np.random.uniform(*init_distance_bound)])
    #         velocity = np.array([np.random.uniform(*init_velocity_bound)])

    #         ego_pos, obstacle_pos = self.generate_vehicles_position(distance)

    #         self.spawn_obstacle_vehicle(obstacle_pos)
    #         self.spawn_ego_vehicle(ego_pos)

    #         if self.ego_vehicle is not None:
    #             break

    #     self.setup_sensors()
    #     for i in range(10):
    #         if not self.disable_rendering:
    #             snapshot, image = self.tick(timeout=2.0)
    #         else:
    #             snapshot = self.tick(timeout=2.0)[0]
    #     if not self.disable_rendering:
    #         #image.save_to_disk("{}.png".format(str(self.step_num).zfill(3)))
    #         self.step_num += 1 
    #         self.image = self._get_image_raw_data(image)

    #     self.set_ego_vehicle_velocity(velocity)

    #     #if not self.disable_rendering:
    #     #    snapshot, image = self.tick(timeout=2.0)
    #     #    self.image = self._get_image_raw_data(image)
    #     #else:
    #     #    snapshot = self.tick(timeout=2.0)[0]

    #     if not self.collision_queue.empty():
    #         collision = self.collision_queue.get().normal_impulse
    #         is_collision = True
    #         raise Exception("Collision happened at the beginning of the episode")
    #     else:
    #         is_collision = False

    #     if self.image_observation:
    #         obs = {'velocity': velocity.item(), 'distance': distance.item()}
    #         obs['image'] = self.image
    #     else:
    #         obs = np.array([distance.item(), velocity.item()], dtype=np.float32)
    #     self.action_state = 0.0
    #     return obs  # reward, done, info can't be included

    def reset(self, state=None):
        """
        Reset env.

        state:
            None -> random reset (original behavior)
            [distance, velocity] -> reset to the specified physical state
        """
        self._destroy_all_actors()
        self.step_num = 0

        if state is None:
            init_distance_bound = [45.0, 60.0]
            init_velocity_bound = [10.0, 25.0]

            while True:
                distance = np.array([np.random.uniform(*init_distance_bound)], dtype=np.float32)
                velocity = np.array([np.random.uniform(*init_velocity_bound)], dtype=np.float32)

                ego_pos, obstacle_pos = self.generate_vehicles_position(distance)

                self.spawn_obstacle_vehicle(obstacle_pos)
                self.spawn_ego_vehicle(ego_pos)

                if self.ego_vehicle is not None and self.obstacle_vehicle is not None:
                    break
                else:
                    self._destroy_all_actors()
                    time.sleep(10.0)
        else:
            distance = np.array([float(state[0])], dtype=np.float32)
            velocity = np.array([float(state[1])], dtype=np.float32)

            while True:
                ego_pos, obstacle_pos = self.generate_vehicles_position(distance)

                self.spawn_obstacle_vehicle(obstacle_pos)
                self.spawn_ego_vehicle(ego_pos)

                if self.ego_vehicle is not None:
                    break
                else:
                    self._destroy_all_actors()

        self.setup_sensors()

        for _ in range(10):
            if not self.disable_rendering:
                snapshot, image = self.tick(timeout=10.0)   # was 2.0
            else:
                snapshot = self.tick(timeout=10.0)[0]       # was 2.0

        # Set initial velocity first
        self.set_ego_vehicle_velocity(velocity)

        # Tick a few more times so velocity + camera image are consistent
        for _ in range(3):
            if not self.disable_rendering:
                snapshot, image = self.tick(timeout=10.0)
            else:
                snapshot = self.tick(timeout=10.0)[0]

        if not self.disable_rendering:
            self.step_num += 1
            self.image = self._get_image_raw_data(image)

        # Drain any collision events that occurred during warm-up ticks.
        # At close distances the vehicles may nudge on spawn; these are
        # not real operational collisions so we discard them here.
        while not self.collision_queue.empty():
            self.collision_queue.get()

        # Use measured values from simulator for consistency
        distance_val = self._compute_distance()
        velocity_val = self._compute_velocity()

        if self.image_observation:
            obs = {
                'velocity': velocity_val,
                'distance': distance_val,
                'image': self.image
            }
        else:
            obs = np.array([distance_val, velocity_val], dtype=np.float32)

        self.action_state = 0.0
        return obs
    
    def reset_to_state(self, distance, velocity):
        return self.reset(state=[distance, velocity])

    def _destroy_all_actors(self):
        for i, _ in enumerate(self.actors):
            if self.actors[i] is not None:
                try:
                    self.actors[i].destroy()
                except Exception:
                    pass
                self.actors[i] = None
        self.actors = []

    def render(self, mode="human"):
        if self.disable_rendering:
            raise Exception("Rendering is disabled, \
                            please set disable_rendering to False")
        
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, please install it with `pip install pygame`."
                )
        
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.surfarray.make_surface(self.image.swapaxes(0, 1))            
        self.screen.blit(self.surf, (0, 0))
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.array3d(self.screen)), (1, 0, 2)
            )
        

    # def close(self):
    #     for i, _ in enumerate(self.actors):
    #         if self.actors[i] is not None:
    #             self.actors[i].destroy()
    #     self.carla_server.stop()
    #     if self.screen is not None:
    #         import pygame

    #         pygame.display.quit()
    #         pygame.quit()

    def close(self):
        self._destroy_all_actors()
        if self.carla_server is not None:
            self.carla_server.stop()
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


class ResizeObservation(ObservationWrapper):

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space['image'].shape[2:]
        self.observation_space['image'] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.observation_space['velocity'] = Box(low=0, high=1.0, shape=(1,), dtype=np.float32)

    def observation(self, observation):
        from PIL import Image

        img = Image.fromarray(observation['image'])
        img = img.crop((250, 250, 550, 550))
        img = img.resize(self.shape[::-1])
        observation['image'] = np.array(img)
        observation['velocity'] = np.array([observation['velocity'] / 30.0])
        return observation


class NormalizeObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                     high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

    def observation(self, observation):
        assert observation.shape == (2,)
        return np.array([observation[0] / 60.0, observation[1] / 30.0])


def advanced_emergency_braking_system_env():
    return NormalizeObservation(CarlaAEBSEnv(disable_rendering=True))


def advanced_emergency_braking_system_env_with_rendering():
    return NormalizeObservation(
        CarlaAEBSEnv(disable_rendering=False, image_observation=False, use_scenic=False)
    )


def vision_advanced_emergency_braking_system_env():
    return ResizeObservation(
        CarlaAEBSEnv(disable_rendering=False, image_observation=True, use_scenic=False),
        shape=(64, 64),
    )


def _register_carla_envs():
    from gym.envs.registration import register

    try:  # gym <= 0.21 keeps specs in registry.env_specs, newer gym in registry
        existing = set(gym.envs.registry.env_specs)
    except AttributeError:
        existing = set(gym.envs.registry)

    for env_id, factory in [
        ('AdvancedEmergencyBrakingSystem-v0', 'advanced_emergency_braking_system_env'),
        ('AdvancedEmergencyBrakingSystemWithRendering-v0',
         'advanced_emergency_braking_system_env_with_rendering'),
        ('VisionAdvancedEmergencyBrakingSystem-v0',
         'vision_advanced_emergency_braking_system_env'),
    ]:
        if env_id not in existing:
            register(env_id, entry_point='env:' + factory, max_episode_steps=300)


_register_carla_envs()
