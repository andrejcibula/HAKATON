# debug_obs.py
from metadrive import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": False,
    "num_scenarios": 5,
    "start_seed": 42,
    "sensors": {"lidar": {}},
    "image_observation": False,
})

obs, info = env.reset()
print("OBS TYPE:", type(obs))
print("OBS:", obs if not hasattr(obs, 'shape') else f"array shape={obs.shape}, min={obs.min():.3f}, max={obs.max():.3f}")
print("INFO TYPE:", type(info))
print("INFO KEYS:", info.keys() if hasattr(info, 'keys') else info)

action = env.action_space.sample()
result = env.step(action)
print("\nSTEP returns", len(result), "values")
obs2, reward, term, trunc, info2 = result
print("STEP OBS TYPE:", type(obs2))
print("STEP INFO KEYS:", info2.keys() if hasattr(info2, 'keys') else info2)

env.close()