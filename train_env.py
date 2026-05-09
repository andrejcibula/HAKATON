import gymnasium as gym
import numpy as np
from metadrive import MetaDriveEnv


class MetaDriveGymEnv(gym.Env):
    """
    Wrapper koji pretvara MetaDrive u standardni Gymnasium env.
    MetaDrive 0.4.3 vraca obs kao flat numpy array od 259 vrednosti.
    Info dict sadrzi: crash, out_of_road, route_completion, itd.
    """

    OBS_SIZE = 259  # MetaDrive 0.4.3 default

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
        """Cistimo obs — zameni NaN, osiguramo shape."""
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.5)
        obs = np.clip(obs, 0.0, 1.0)
        if obs.shape[0] != self.OBS_SIZE:
            # Padding ili rezanje ako se velicina razlikuje
            if obs.shape[0] < self.OBS_SIZE:
                obs = np.pad(obs, (0, self.OBS_SIZE - obs.shape[0]), constant_values=0.5)
            else:
                obs = obs[:self.OBS_SIZE]
        return obs

    def _compute_reward(self, obs, info, action):
        reward = 0.0

        # 1. Napredak po ruti (najvazniji signal)
        route_completion = float(info.get("route_completion", 0.0))
        progress = route_completion - self.prev_route_completion
        reward += progress * 20.0
        self.prev_route_completion = route_completion

        # 2. Brzina — nagradi kretanje, kazni stajanje
        velocity = float(info.get("velocity", 0.0))
        speed_norm = min(velocity / 40.0, 1.0)
        reward += speed_norm * 0.3

        # 3. Glatkoća — kazni nagle promene (jerk)
        action_delta = np.abs(action - self.prev_action)
        reward -= float(np.sum(action_delta)) * 0.3

        # 4. Izletanje sa puta
        if info.get("out_of_road", False):
            reward -= 10.0

        # 5. Sudar — najveci penal
        if info.get("crash", False):
            reward -= 25.0

        # 6. Stigao do cilja — bonus
        if info.get("arrive_dest", False):
            reward += 50.0

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
        reward = self._compute_reward(obs, info, action)

        self.prev_action = action.copy()
        self.steps += 1

        # Kraj epizode ako je sudar ili izletanje
        if info.get("crash", False) or info.get("out_of_road", False):
            terminated = True

        # Max duzina epizode
        if self.steps >= 2000:
            truncated = True

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()