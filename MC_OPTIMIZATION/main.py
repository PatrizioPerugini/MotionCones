import numpy as np
from mc_optimizer import MotionPlanner
from utils import plot_solution

def main():
    N_steps = 4#7
    start_position = np.array([0.02, -0.02, 0.0])
    goal_position = np.array([0.01, 0.01, -0.2])
    planner = MotionPlanner(N_steps = N_steps ,initial_state=start_position, goal_state=goal_position)
    initial_guess = planner.get_ig()
    print(initial_guess)
    planner.callback_func(initial_guess)
    print("starting optimization")
    #T, velocities = planner.optimize_motion_cone(N, initial_guess, start_position, goal_position)
    T, velocities, optimized_state = planner.optimize_motion_cone()
    
    print("trajectory time is: ", T)
    print("\nvelocities are: ", velocities)

    plot_solution(optimized_state=optimized_state, goal_position=goal_position)



if __name__=='__main__':
    main()