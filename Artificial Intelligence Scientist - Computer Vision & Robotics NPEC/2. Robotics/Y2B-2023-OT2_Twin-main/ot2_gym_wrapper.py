import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render_enabled = render
        self.max_steps = max_steps
        self.prev_distance = 0.0
        self.consecutive_right_direction = 0
        self.consecutive_wrong_direction = 0

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32) # YOUR CODE HERE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32) # YOUR CODE HERE

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = np.random.uniform(low=[-0.260, -0.260, 0.080], high=[0.180, 0.130, 0.200])# YOUR CODE HERE
        # self.goal_position = np.random.uniform(low=[-0.1872, -0.1707,  0.1196], high=[0.253,  0.2195, 0.2895])# YOUR CODE HERE
        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        # print("observation::",observation)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        robot_id = next(iter(observation))  # Get the first key in the dictionary
        pipette_position = observation[robot_id]['pipette_position']
        observation = np.concatenate([np.array(pipette_position), self.goal_position], dtype=np.float32)

        # Reset the number of steps
        self.steps = 0

        return observation, {}
    

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.concatenate([action, [0]]) # YOUR CODE HERE

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        robot_id = next(iter(observation))  # Get the first key in the dictionary
        pipette_position = observation[robot_id]['pipette_position']
        observation = np.concatenate([np.array(pipette_position), self.goal_position], dtype=np.float32) # YOUR CODE HERE

        #newnew

        pipette_position = np.array(pipette_position)
        self.goal_position = np.array(self.goal_position)

        current_distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        prev_distance_to_goal = np.linalg.norm(self.prev_pipette_position - self.goal_position) if hasattr(self, 'prev_pipette_position') and self.prev_pipette_position is not None else current_distance_to_goal

        # Calculate improvement in distance
        distance_improvement = prev_distance_to_goal - current_distance_to_goal

        # Define constants for rewards, penalties, and scaling factors
        GOAL_REACHED_REWARD = 2000.0
        WRONG_DIRECTION_PENALTY = 10

        # Initialize static penalty for each time step
        time_step_penalty = -0.01

        # Update penalty/reward based on the direction of movement
        if distance_improvement > 0:  # Moving towards the goal
            self.consecutive_wrong_direction = 0
            self.consecutive_right_direction += 1
            distance_reward = distance_improvement * self.consecutive_right_direction
        else:  # Moving away from the goal
            self.consecutive_wrong_direction += 1
            self.consecutive_right_direction = 0
            distance_reward = distance_improvement * self.consecutive_wrong_direction * WRONG_DIRECTION_PENALTY
            # Increase the penalty for each consecutive step in the wrong direction

        # Check if the goal is reached
        termination_threshold = 0.001
        if current_distance_to_goal < termination_threshold:
            terminated = True
            reward = GOAL_REACHED_REWARD
        else:
            terminated = False
            reward = distance_reward + time_step_penalty

        # Check for truncation
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Update the previous position for the next step
        self.prev_pipette_position = pipette_position.copy()



        #new
        
        # pipette_position = np.array(pipette_position)
        # self.goal_position = np.array(self.goal_position)

        # current_distance = np.linalg.norm(pipette_position - self.goal_position)

        # distance_change = self.prev_distance - current_distance

        # # Update previous distance for the next step
        # self.prev_distance = current_distance

        # # Calculate reward based on current distance and change in distance
        # distance_reward = -current_distance
        # change_reward = distance_change  # You may want to scale this if needed

        # # Combine rewards (you can adjust the coefficients based on your preference)
        # reward = 0.7 * distance_reward + 0.3 * change_reward

        # # # Check if the task is complete
        # # if distance_change > 0:
        # #     terminated = True
            
        # #     reward += 0.5 * reward
        # # else:
        # #     terminated = False
       

        # completion_threshold = 0.001
        # if current_distance < completion_threshold:
        #     terminated = True
        #     # Increase the reward by 50% if the goal is reached
        #     reward += 0.5 * reward
        # else:
        #     terminated = False

        # # Check if the episode should be truncated
        # if self.steps >= self.max_steps:
        #     truncated = True
        # else:
        #     truncated = False

        info = {}  # No additional information to return

        # Increment the number of steps
        self.steps += 1

        # end new

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        # reward = float(-np.linalg.norm(observation[:3] - observation[3:])) # YOUR CODE HERE
        
        # # next we need to check if the if the task has been completed and if the episode should be terminated
        # # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        # completion_threshold = 0.01
        # robot_id = next(iter(observation))  # Get the first key in the dictionary

        # # Check completion based on the distance between the pipette position and the goal position
        # if np.linalg.norm(observation[:3] - self.goal_position) < completion_threshold: # YOUR CODE HERE:
        #     terminated = True
        #     # we can also give the agent a positive reward for completing the task
        #     reward += 1.0
        # else:
        #     terminated = False

        # # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        # if self.steps >= self.max_steps :# YOUR CODE HERE:
        #     truncated = True
        # else:
        #     truncated = False

        # info = {} # we don't need to return any additional information

        # print(f'terminated:{terminated}, truncated: {truncated}')

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()




