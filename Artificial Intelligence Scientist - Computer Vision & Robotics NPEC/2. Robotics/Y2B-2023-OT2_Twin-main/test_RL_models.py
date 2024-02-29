import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

num_agents = 1
env = OT2Env(render=True)
obs, info = env.reset()

model = PPO.load(r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\models\new_best\model.zip")

goal_pos = np.array([0.13, 0.16, 0.12])  
env.goal_position = goal_pos

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)

    obs[3:] = goal_pos

    print(obs)

   
    distance = obs[3:] - obs[:3]  

    
    error = np.linalg.norm(distance)
    print("Error:", error)

    
    if error < 0.01:  
        action = np.array([0, 0, 0, 1])  
        obs, rewards, terminated, truncated, info = env.step(action)
        break

    if terminated:
        obs, info = env.reset()