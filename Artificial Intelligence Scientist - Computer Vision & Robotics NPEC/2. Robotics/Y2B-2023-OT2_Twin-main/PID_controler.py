import gymnasium as gym
from simple_pid import PID
from math import atan2, degrees, pi
from ot2_gym_wrapper import OT2Env
import pybullet

pid_controller_x = PID(Kp=5, Ki=0.5, Kd=2)
pid_controller_y = PID(Kp=5, Ki=0.5, Kd=2)
pid_controller_z = PID(Kp=5, Ki=0.5, Kd=2)



desired_x_position = 1
desired_y_position = 1
desired_z_position = 1


pid_controller_x.setpoint = desired_x_position
pid_controller_y.setpoint = desired_y_position
pid_controller_z.setpoint = desired_z_position


env = OT2Env(render = True)

observation, info = env.reset()

for i in range(2000):
    
    current_x_position, current_y_position, current_z_position = observation[0:3]

    print("obserwation: ",observation[0:3])

    print(desired_x_position,desired_y_position,desired_z_position)

    
    action_x = pid_controller_x(current_x_position)
    action_y = pid_controller_y(current_y_position)
    action_z = pid_controller_z(current_z_position)

    action = [action_x, action_y, action_z]

    print("action", action)

    
    observation, reward, terminated, truncated, info = env.step(action)

    

    
    env.render()
    

