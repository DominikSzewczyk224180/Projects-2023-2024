from sim_class import Simulation

sim = Simulation(num_agents=1)  

corner_patterns = [
    [1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, -1, 1, 1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [-1, 1, -1, 1],
    [-1, -1, -1, 1],
    [1, -1, -1, 1]
]


corner_coordinates = []

for pattern in corner_patterns:
   
    for _ in range(200):
        actions = [pattern]
        sim.run(actions)

    
    state = sim.run(actions)

    
    robot_position = state.get('robotId_1', {}).get('robot_position', None)
    if robot_position is not None:
        corner_coordinates.append(robot_position)
        print(f"Recorded position for corner: {robot_position} (Velocity pattern: {pattern[:3]}, Drop command: {pattern[3]})")
    else:
        print("Position information not available.")


print("\nCoordinates of the working envelope corners:")
for i, corner_coordinate in enumerate(corner_coordinates, 1):
    print(f"Corner {i}: {corner_coordinate}")





