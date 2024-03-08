from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from model import Model
import sympy as sp
from algorithms.rrt import RRT
from utils import plot_moving_square_with_rotation, edit_video



#goal_states = [[-0.2,-0.3,0],[0.2,-0.3,0],[0.2,0.3,0],[-0.2,0.3,0]]#it works with g, the other sign of y works with -g
#goal_states = [[0.01,0.02,0],[-0.01,0.02,0] ,[-0.,-0.02,np.pi/14],[-0.01,-0.02,0],[0.01,-0.02,0],[0,0.02,0]]#,[0.1,0.,0],[-0.1,0.,0] ]
goal_states = [[-0.01,0.02,0.3]]
#goal_states = [[0.1,0.2,np.pi/10],[-0.1,0.2,0] ,[-0.,-0.2,0],[-0.1,-0.2,0],[0.1,-0.2,0],[0,0.2,0]]
results = []
push = []
paths = []
for gs in goal_states:
    print("STARTING NEW GOAL PATH WITH GOAL: ",  gs)
    m = Model(goal_state=gs)
    planner = RRT(m)
    seed = np.random.randint(0,10**6)
    np.random.seed(seed)

    goal, plan,node_list = planner.plan(max_nodes=90)
    results.append(goal)
    pushers = [ m.wall_orientations[node.from_rs] for node in plan if node.from_rs is not None ]
    path = [(node.state[0], node.state[1]) for node in plan]
    push.append(pushers)
    paths.append(path)
print("goals where found if 1: \n",results)
for i in range(len(results)):
    print("goal found? ", results[i])
    print("goal:       ", goal_states[i])
    print("push found  ",push[i])
    print("path found  ",paths[i])
    print("--------------\n")

#plan = [[0.5, 0.,  0. ], [ 0.22048551,  0.23625803, -0.01319548], [ 0.44097103,  0.47251606, -0.02639097]]
#[[0 0 0]], [[-0.08352788  0.033137   -0.01384354]], [[-0.10191789  0.0990026  -0.02780613]], [[-0.13157865  0.16999341 -0.04174918]], [[-0.19075331  0.2383179  -0.05564204]]]
#print("GOAL FOUND: ", goal)
'''
plot_list = np.array(node_list)
if plan is not None:
    print("The PLAN is: \n ", plan)
    r = m.r
    path = [(node.state[0], node.state[1]) for node in plan]
    angles = [node.state[2] for node in plan]
    pushers = [node.from_rs for node in plan if node.from_rs is not None]
    print("pushers used: ", pushers)

    pushers.append(0)
    gt = m.goal_state.tolist()
    plot_moving_square_with_rotation(path,angles,pushers,gt, dir="./animation_frames",circular_object_radius=r)
    edit_video("./animation_frames",len(path),0.1, speed=0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of points
    ax.scatter(plot_list[:, 0], plot_list[:, 1], plot_list[:, 2])
    
    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Show the plot
    plt.show()

    print('n states: ',len(node_list))
    

else:

    print("Unable to find path")
    '''