from sim_class import Simulation
import random
# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=2)  # For two robots

for i in range(1000):
    velocity_x = random.uniform(-0.5, 0.5)
    velocity_y = random.uniform(-0.5, 0.5)
    velocity_z = random.uniform(-0.5, 0.5)
    drop_command = random.randint(0, 1)

    actions = [[velocity_x, velocity_y, velocity_z, drop_command],
               [velocity_x, velocity_y, velocity_z, drop_command]]
    
    sim.run(actions)
    state = sim.run(actions)
    print(state)

# sim.reset(num_agents=1)