import gymnasium as gym
import time
import stable_baselines3 as sb3
from stable_baselines3 import PPO

env = gym.make('Pendulum-v1', render_mode='human',g=2) 
model = PPO.load('\pendulum_models\pendulum16.zip')

obs, info = env.reset()

for _ in range(1000):   

    action = model.predict(obs, deterministic= True )[0]
    
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    
