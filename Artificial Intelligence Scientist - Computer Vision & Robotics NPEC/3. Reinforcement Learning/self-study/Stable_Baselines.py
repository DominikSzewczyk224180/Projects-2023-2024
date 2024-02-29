import gymnasium as gym
import time
import stable_baselines3 as sb3
from stable_baselines3 import PPO

env = gym.make('Pendulum-v1', render_mode='rgb_array',g=2) 
model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000,progress_bar=True)

for i in range(100):
    model.learn(total_timesteps=10000,progress_bar=True)
    model.save(f"./pendulum_models/pendulum{i}")


# observation, info = env.reset() 

# for _ in range(1000):   
 
#     action = env.action_space.sample()
    
#     observation, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     env.render()



