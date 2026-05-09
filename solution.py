import numpy as np

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class Solution:

<<<<<<< Updated upstream
    OBS_SIZE = 259
=======
    OBS_SIZE = 9  # 7 lidar sektora + lateral_left + lateral_right
>>>>>>> Stashed changes

    def __init__(self, game):
        self._game = game
        self.model = None
        self.prev_action = np.zeros(2, dtype=np.float32)
<<<<<<< Updated upstream
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
=======

        if SB3_AVAILABLE:
            try:
                self.model = PPO.load("best_model/best_model", device="cpu")
                print("[Solution] Model ucitan!")
            except Exception:
                print("[Solution] Model nije pronadjen — rule-based fallback")
>>>>>>> Stashed changes

    @property
    def config(self):
        return {
            "image_observation": False,
<<<<<<< Updated upstream
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
=======
        }

    def _get_obs(self, simulator_output):
        obs = simulator_output.get("observation", None)
        if obs is None:
            return np.ones(self.OBS_SIZE, dtype=np.float32) * 0.5

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

        return np.array([
            napred, levo_n1, levo_n2, levo, desno, desno_n2, desno_n1,
            lateral_left, lateral_right
        ], dtype=np.float32)

    def do_iteration(self, simulator_output, user_input=None):
        if user_input is not None:
            if isinstance(user_input, (list, tuple)) and len(user_input) == 2:
                us, ut = float(user_input[0]), float(user_input[1])
                if abs(us) > 0.05 or abs(ut) > 0.05:
                    return [float(np.clip(us, -1, 1)), float(np.clip(ut, -1, 1))]

        if self.model is None:
            return [0.0, 0.3]

        obs = self._get_obs(simulator_output)
        action, _ = self.model.predict(obs, deterministic=True)

>>>>>>> Stashed changes
        s_delta = np.clip(action[0] - self.prev_action[0], -0.08, 0.08)
        action[0] = self.prev_action[0] + s_delta
        self.prev_action = action.copy()

<<<<<<< Updated upstream
        # Korisnicki input override
        if user_input is not None:
            us = user_input.get("steering", 0.0)
            ut = user_input.get("throttle", 0.0)
            if abs(us) > 0.05 or abs(ut) > 0.05:
                action[0] = us * 0.7 + action[0] * 0.3
                action[1] = ut

=======
>>>>>>> Stashed changes
        return [
            float(np.clip(action[0], -1.0, 1.0)),
            float(np.clip(action[1], -1.0, 1.0))
        ]