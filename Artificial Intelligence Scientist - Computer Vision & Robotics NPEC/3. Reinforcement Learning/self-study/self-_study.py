import gymnasium as gym
import time

# Create an instance of the CartPole environment
# Set render_mode to 'human' to render the environment in a window
env = gym.make('Pendulum-v1', render_mode='rgb_array') # Set render_mode to 'rgb_array' to render the environment as an image array

# Reset the environment and get the initial state
observation, info = env.reset(seed=42) # Set seed to get the same initial state every time

# Run the simulation for 1000 steps
for _ in range(1000):   
    #if you want to render the environment as an image array use the following code
    #img = env.render() # This will store the image array in the variable img instead of rendering it in a window

    # Take a random action by sampling from the action space
    action = env.action_space.sample()
    
    # Execute the action and get the next state, reward, and whether the episode is done. 
    # Terminated is True if the episode is done and False otherwise, Truncated is True if the episode was terminated because the time limit was reached.
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # add a delay to slow down the env render
    # This is purely for visualization purposes, DO NOT use this when training!
    time.sleep(0.05)
    
    # If the episode is done, reset the environment
    if done:
        state = env.reset()