import gymnasium as gym
import numpy as np
from metadrive import MetaDriveEnv


class MetaDriveGymEnv(gym.Env):

    OBS_SIZE = 9

    def __init__(self, config=None):
        super().__init__()

        self.env_config = {
            "use_render": False,
            "num_scenarios": 100,
            "start_seed": 42,
            "traffic_density": 0.1,
            "map": "SSSSSSSSS",
            "image_observation": False,
        }
        if config:
            self.env_config.update(config)

        self.env = MetaDriveEnv(self.env_config)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.OBS_SIZE,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_route_completion = 0.0
        self.steps = 0

    def _process_obs(self, obs):
        full = np.array(obs, dtype=np.float32)
        full = np.nan_to_num(full, nan=0.5)
        lidar = full[:240]

        napred   = min(np.min(lidar[0:20]), np.min(lidar[220:240]))
        levo_n1  = np.min(lidar[20:50])
        levo_n2  = np.min(lidar[50:90])
        levo     = np.min(lidar[90:120])
        desno    = np.min(lidar[120:150])
        desno_n2 = np.min(lidar[150:190])
        desno_n1 = np.min(lidar[190:220])

        lateral_left  = np.clip(full[240] if len(full) > 240 else 0.5, 0.0, 1.0)
        lateral_right = np.clip(full[241] if len(full) > 241 else 0.5, 0.0, 1.0)

        return np.clip(np.array([
            napred, levo_n1, levo_n2, levo, desno, desno_n2, desno_n1, lateral_left, lateral_right
        ], dtype=np.float32), 0.0, 1.0)

    def _compute_reward(self, info, action):
        reward = 0.0
    
        route_completion = float(info.get("route_completion", 0.0))
        nav_forward = float(info.get("navigation_forward", 0.0)) 
        nav_left  = float(info.get("navigation_left", 0.0))
        nav_right = float(info.get("navigation_right", 0.0))    
        progress = route_completion - self.prev_route_completion
        reward += progress * 10.0
        self.prev_route_completion = route_completion
        lateral_left  = float(info.get("lateral_to_left", 0.5))
        lateral_right = float(info.get("lateral_to_right", 0.5))
        total = lateral_left + lateral_right

        step_distance = float(info.get("step_reward", 0.0))
        reward += step_distance * 0.2

        velocity = float(info.get("velocity", 0.0))
        speed_norm = min(velocity / 30.0, 1.0) * 0.5
        #reward += speed_norm * 1.0
        reward += nav_forward * 0.2


        if total > 0:
            # Koliko je centriran — 1.0 = savrsena sredina, 0.0 = na ivici
            center_score = 1.0 - abs(lateral_left - lateral_right) / total
            reward += center_score * 0.5 #bilo 1.5
        if lateral_left < 0.2 or lateral_right < 0.2:
            reward -= 1.0
        if velocity < 0.5: #bilo 1
            reward -= 0.5

        if nav_forward < 0.3:
            reward -= 1.0

        action_delta = np.abs(action - self.prev_action)
        reward -= float(np.sum(action_delta)) * 0.1

        if info.get("out_of_road", False):
            reward -= 15.0*5.0

        if info.get("crash", False):
            reward -= 30.0

        if info.get("arrive_dest", False):
            reward += 100.0

        return reward

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset()
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_route_completion = 0.0
        self.steps = 0
        return self._process_obs(raw_obs), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        raw_obs, _, terminated, truncated, info = self.env.step(action)

        obs = self._process_obs(raw_obs)
        reward = self._compute_reward(info, action)

        self.prev_action = action.copy()
        self.steps += 1

        if info.get("crash", False) or info.get("out_of_road", False):
            terminated = True

        if self.steps >= 3000:
            truncated = True

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()