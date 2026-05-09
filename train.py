import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from train_env import MetaDriveGymEnv

print("=== MetaDrive PPO Trening ===")

env = Monitor(MetaDriveGymEnv())

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("best_model", exist_ok=True)

checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="ppo_metadrive"
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048, #bilo 2048
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [32, 16]},
    verbose=1,
    device="cpu"
)
#model = PPO.load("best_model/best_model", env=env, device="cpu")

print("\nPocinjem trening (50,000 koraka)...")

model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_cb],
    progress_bar=True
)
"""model.learn(
    total_timesteps=10_000,
    callback=[checkpoint_cb],
    progress_bar=True,
    reset_num_timesteps=False
)"""

model.save("best_model/best_model")
print("\nTrening zavrsen! Fajl: best_model/best_model.zip")