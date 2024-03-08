from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, LinearConstraint
import cv2

#If I am getting closer to the node I can add it, otherwise it makes no sense.
def transition_test(q_parent, q_sample, q_goal, treshold = 0.):
    #print("parent: ", q_parent)
    #print("sample: ", q_sample)
    #print("goal: ", q_goal)
    #print("dist pre: ", np.linalg.norm(q_parent-q_goal))
    #print("dist post: ", np.linalg.norm(q_sample-q_goal))
    #input("orcoddi")
    pre_distance = np.linalg.norm(q_parent-q_goal)
    post_distance = np.linalg.norm(q_sample-q_goal)
    
    return post_distance+treshold<pre_distance



def plot_convex_hull(points , hull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')
    
    # Plot the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the simplex
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')
        
    

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show the plot
    plt.show()

def plot_hull_and_points(hull, point, projected_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot convex hull
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='original point')
    ax.scatter(projected_point[0], projected_point[1], projected_point[2], color='green', s = 7, label='projected point')

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def objective(x, original_point):
    return np.sum((x - original_point)**2)

def find_closest_projection(hull, original_point):
    hull_equations = hull.equations[:, :3]
    hull_biases = -hull.equations[:, 3]  # Negate biases for inequalities

    # Set up linear inequality constraints Ax <= b
    A = hull_equations
    b = hull_biases


    bounds = [(None, None)] * 3  # Replace with your desired bounds

    # Set up linear inequality constraints
    linear_constraint = LinearConstraint(A, -np.inf, b, keep_feasible=False)

    # Run the optimization
    result = minimize(objective, original_point, args = (original_point, ), constraints=[linear_constraint], bounds=bounds)

    # The optimal point is the projection onto the convex hull
    projected_point = result.x

    distance = np.linalg.norm(original_point - projected_point)

    return projected_point, distance




def project_onto_convex_hull(hull, point):

    #int_points = sample_points_inside_convex_hull(hull,1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Plot convex hull
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='Point to Test')
   # ax.scatter(*int_points, color='pink', s=0.1)
    
    # label='Point to Test')

    # Find and plot the closest projection
    closest_projection, dist = find_closest_projection(hull, point)
    ax.plot([point[0], closest_projection[0]], [point[1], closest_projection[1]], [point[2], closest_projection[2]], "c<:")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()





################ PROJECTION STUFF

def plot_hull_and_point(hull, point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    proj_point = project_onto_convex_hull(hull, point)

    # Plot convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='Point to Test')
    ax.scatter(*proj_point, color='green', label='Point to proj')


    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def test_hull(): # Example usage
    points = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    hull = ConvexHull(points)

    # Test points
    test_point_inside = np.array([0.5, 0.5])
    test_point_outside = np.array([2, 2])

    # Check if points are inside the convex hull
    result_inside = point_in_hull(test_point_inside, hull)
    result_outside = point_in_hull(test_point_outside, hull)

    print(f"Is {test_point_inside} inside the convex hull? {result_inside}")
    print(f"Is {test_point_outside} inside the convex hull? {result_outside}")

def plot_moving_square_with_rotation(path, rotation_angles,pushers, gt = [-0.1,0.2,-np.pi/8],circular_object_radius=0.2, dir="./"):
    gt_x , gt_z, gt_theta = gt
    print(pushers)
    print(gt)
    for i in range(len(path)):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        wall = pushers[i]
        # Plot the circular object (end effector) at the fixed position (0, 0)
        circular_object_theta = np.linspace(0, 2 * np.pi, 100)
        circular_object_x = circular_object_radius * np.cos(circular_object_theta)
        circular_object_y = circular_object_radius * np.sin(circular_object_theta)
        ax.plot(circular_object_x, circular_object_y, color='red', label='End Effector (Fixed)')

        state_x, state_z = path[i]
        rotation_angle = rotation_angles[i] 

        # Rotate the square object and the wall around the y-axis
        rotation_matrix_y = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                      [0, 1, 0],
                                      [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

        rotation_matrix_y_GT = np.array([[np.cos(gt_theta), 0, np.sin(gt_theta)],
                                      [0, 1, 0],
                                      [-np.sin(gt_theta), 0, np.cos(gt_theta)]])

        #square_vertices = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
        square_vertices = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],[-0.5, 0.5]])
        rotated_square = np.dot(np.column_stack([square_vertices, np.zeros((5, 1))]),
                                rotation_matrix_y.T)[:, :2] + np.array([state_x, state_z])

        rotated_square_GT = np.dot(np.column_stack([square_vertices, np.zeros((5, 1))]),
                                rotation_matrix_y_GT.T)[:, :2] + np.array([gt_x, gt_z])

        # Plot the rotated square object
        ax.plot(rotated_square[:, 0], rotated_square[:, 1], color='blue', label='Moving Tilted Object')

        ax.plot(rotated_square_GT[:, 0], rotated_square_GT[:, 1], color='green', label='GT pose')

        ### Plot the side of the square initially in contact with the wall
        ##ax.plot([rotated_square[0, 0], rotated_square[3, 0]], [rotated_square[0, 1], rotated_square[3, 1]],
        ##        color='blue', linestyle='-', label='Initial Wall Side')


           ##if pusher below
        rotation_pusher = 0#-np.pi/2
      
        rotation_matrix_y_pusher = np.array([[np.cos(rotation_pusher), -np.sin(rotation_pusher), 0],
                                     [np.sin(rotation_pusher), np.cos(rotation_pusher), 0],
                                     [0, 0, 1]])

        rot_below = rotation_matrix_y_pusher@rotation_matrix_y
        
        wall_height = 1  # Adjust this value to set the height of the wall
        # if orizontal wall
        if wall == 0 :
            initial_wall_start = np.array([ state_x - 0.5 , state_z-0.5, 0])
            initial_wall_end = np.array([ state_x + wall_height - 0.5 , state_z-0.5, 0])      
            rotated_wall_start = np.dot(initial_wall_start, rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot(initial_wall_end, rotation_matrix_y.T)[:2]

        # Plot the rotated wall as a vertical line
        #elif vertical wall:
        elif wall == 2:
            rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rot_below.T)[:2]
            rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rot_below.T)[:2]
        else:
            rotated_wall_start = np.dot([-0.5 + state_x, state_z - wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot([-0.5 + state_x, state_z + wall_height / 2, 0], rotation_matrix_y.T)[:2]
            #rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rot_below.T)[:2]
            #rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rot_below.T)[:2]


        ##

        ax.plot([rotated_wall_start[0], rotated_wall_end[0]], [rotated_wall_start[1], rotated_wall_end[1]],
                color='red', linestyle='-', linewidth=8, label='Rotated Wall')
                

        rotated_wall_start_gt = np.dot([ gt_x - 0.5 , gt_z-0.5, 0], rotation_matrix_y_GT.T)[:2]
        rotated_wall_end_gt = np.dot([ gt_x + wall_height - 0.5 , gt_z-0.5, 0], rotation_matrix_y_GT.T)[:2] 

        #if vertical wall
        #rotated_wall_start_gt = np.dot([0.5 + gt_x, gt_z - wall_height / 2, 0], rotation_matrix_y_GT.T)[:2]
        #rotated_wall_end_gt = np.dot([0.5 + gt_x, gt_z + wall_height / 2, 0], rotation_matrix_y_GT.T)[:2]
        
        ax.plot([rotated_wall_start_gt[0], rotated_wall_end_gt[0]], [rotated_wall_start_gt[1], rotated_wall_end_gt[1]],
                color='#FF9999', linestyle='-', linewidth=8, label='Rotated Wall_ gt')

    
        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        ax.legend()

        # Save the figure
        fig.savefig(f"{dir}/frame" + str(i) + ".png")

        # Show the plot
        plt.show()

        plt.close()


def edit_video(path,N,dt, speed=1.0):

    img_array = []
    img_path = path + "/frame"
    for n in range(N):
        filename = img_path + str(n)+ '.png'
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fps = int(1/dt)*speed
    out = cv2.VideoWriter(path+'/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return out
#plan = [
#    [0, 0, 0],
#    [-0.0365205, 0.00971316, -0.00343372],
#    [-0.02887251, 0.00988052, -0.0068812],
#    [0.00411625, 0.04963975, -0.01323468],
#    [-0.02669114, 0.03453297, -0.02140957],
#    [-0.03758324, 0.06012168, -0.04124714],
#    [0.00867714, 0.06807457, -0.06604483],
#    [-0.04396994, 0.05483787, -0.09919873],
#    [-0.05593095, 0.09455506, -0.18339331],
#    [-0.0935852, 0.20194937, -0.35798662]
#]
plan = [
    [0, 0, 0],
    [-0.04396994, 0.05483787, -0.09919873],
    [-0.05593095, 0.09455506, -0.0018339331],
    [-0.01, 0.20194937, 0]#-0.35798662]
]

pushers = [0, 1, 2, None]

#plan = [[0.5, 0.,  0. ], [ 0.22048551,  0.23625803, -0.01319548], [ 0.44097103,  0.47251606, -0.02639097]]
path = [point[:2] for point in plan]
angles = [point[2] for point in plan]
#i = 0
#for p in plan:
    
#plot_moving_square_with_rotation(path,angles, pushers, gt=[-0.1,0.2,-0], dir="./",circular_object_radius=0.1)    
#i+=1
