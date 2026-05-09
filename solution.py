import numpy as np

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class Solution:

    OBS_SIZE = 259

    def __init__(self, game):
        self._game = game
        self.model = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._load_model()

    def _load_model(self):
        if not SB3_AVAILABLE:
            print("[Solution] SB3 nije instaliran — fallback")
            return

        for path in ["best_model/best_model", "ppo_metadrive_final"]:
            try:
                self.model = PPO.load(path, device="cpu")
                print(f"[Solution] Model ucitan: {path}")
                return
            except Exception:
                continue

        print("[Solution] Model nije pronadjen — rule-based fallback")

    @property
    def config(self):
        return {
            "image_observation": False,
            "sensors": {"lidar": {}},
        }

    def _process_obs(self, obs):
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.5)
        obs = np.clip(obs, 0.0, 1.0)
        if obs.shape[0] != self.OBS_SIZE:
            if obs.shape[0] < self.OBS_SIZE:
                obs = np.pad(obs, (0, self.OBS_SIZE - obs.shape[0]), constant_values=0.5)
            else:
                obs = obs[:self.OBS_SIZE]
        return obs

    def _rule_based_fallback(self, simulator_output):
        obs = np.array(simulator_output, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.5)

        lidar = obs[:240]
        n = len(lidar)

        front = np.min(np.concatenate([lidar[:n//8], lidar[7*n//8:]]))
        left  = np.mean(lidar[n//4:n//2])
        right = np.mean(lidar[n//2:3*n//4])

        throttle = 0.35 if front > 0.3 else -0.4
        steering = np.clip((left - right) * 0.5, -0.6, 0.6)

        s_delta = np.clip(steering - self.prev_action[0], -0.08, 0.08)
        steering = self.prev_action[0] + s_delta
        self.prev_action = np.array([steering, throttle])

        return [float(np.clip(steering, -1, 1)), float(np.clip(throttle, -1, 1))]

    def do_iteration(self, simulator_output, user_input=None):
        if self.model is None:
            return self._rule_based_fallback(simulator_output)

        obs = self._process_obs(simulator_output)
        action, _ = self.model.predict(obs, deterministic=True)

        # Anti-jerk smooth
        s_delta = np.clip(action[0] - self.prev_action[0], -0.08, 0.08)
        action[0] = self.prev_action[0] + s_delta
        self.prev_action = action.copy()

        # Korisnicki input override
        if user_input is not None:
            us = user_input.get("steering", 0.0)
            ut = user_input.get("throttle", 0.0)
            if abs(us) > 0.05 or abs(ut) > 0.05:
                action[0] = us * 0.7 + action[0] * 0.3
                action[1] = ut

        return [
            float(np.clip(action[0], -1.0, 1.0)),
            float(np.clip(action[1], -1.0, 1.0))
        ]