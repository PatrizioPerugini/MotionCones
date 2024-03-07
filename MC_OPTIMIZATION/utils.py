import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def t2v(A):
    v = np.zeros((3, 1))
    v[0:2, 0] = A[0:2, 2]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 0])
    return v

def v2t(v):
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
                  [s, c, v[1]],
                  [0, 0, 1]])
    return A

def box_plus(p_n_x, p_n_z, p_n_w, dx, dz, dw):
    transf_incr = v2t([ dx, dz, dw])
    transf = v2t([ p_n_x, p_n_z, p_n_w])
    update = transf_incr@transf
    res = t2v(update)
    return res 

def box_minus(self, T1, T2):
    """
    Compute the relative transformation between two SE(2) transformations T1 and T2.
    Parameters:
        T1: np.ndarray, shape (3, 3)
            Transformation matrix T1.
        T2: np.ndarray, shape (3, 3)
            Transformation matrix T2.
    Returns:
        np.ndarray, shape (3, 1)
            Vector representing the relative transformation from T1 to T2.
    """
    inv_T1 = np.linalg.inv(T1)
    delta_T = inv_T1 @ T2
    return self.t2v(delta_T)

def check_p_in_cone(self,current_state,current_twist):
    #compute cone
    motion_cones, _ = self.get_motionCones(current_state, theta_rad=0, vel_i= current_twist)
    for equation in motion_cones.equations:
        a = equation[:-1]  # Coefficients a1, a2, ..., an
        b = equation[-1]   # Constant term b
        is_inside = (self.cumput_hp(a, b, current_twist) >=0)
        if not is_inside:
            return False
    return True

def plot_solution(optimized_state, goal_position):
    
    
    arr_x = [coord[0] for coord in optimized_state]
    arr_y = [coord[1] for coord in optimized_state]
    plt.figure(figsize=(10, 6))
    plt.plot(arr_x, arr_y, marker='o', label='Gripper Position')  
    plt.scatter(goal_position[0], goal_position[1], color='r', label='Goal Position')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title('Gripper Trajectory (X-Z Plane)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convex_hull_3d(points, convex_hull_inequalities, point_to_check=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')

    # Plot convex hull faces
    for normal, d in convex_hull_inequalities:
        if normal[2] != 0:  # Check if normal[2] is not zero to avoid division by zero
            xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
            zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            ax.plot_surface(xx, yy, zz, alpha=0.2)

    # Plot convex hull edges
    for i in range(len(points)):
        for j in range(i, len(points)):
            ax.plot([points[i][0]], [points[i][1]], [points[i][2]], 'ro')  # Plot points
            #j = (i + 1) % len(points)
            ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], [points[i][2], points[j][2]], 'g-')  # Plot edges

    # Plot point to check if provided
    if point_to_check is not None:
        ax.scatter(point_to_check[0], point_to_check[1], point_to_check[2], c='r', marker='*', s=100, label='Point to Check')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull')
    ax.legend()
    plt.show()

def orientation_3d(p, q, r):
    """
    Determine the orientation of triplet (p, q, r) in 3D space.
    Returns:
    - 0 if colinear
    - 1 if clockwise
    - 2 if counterclockwise
    """
    #val = np.cross_product(p.subtract(r), q.subtract(r))
    #print("val: ", val)
    val = (q[1] - p[1]) * (r[2] - q[2]) - (q[2] - p[2]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def jarvis_march_3d(points):
    """
    Compute the convex hull in 3D space using the Gift Wrapping (Jarvis March) algorithm.
    Returns:
    - convex_hull_inequalities: List of inequalities representing the half-spaces of the convex hull.
    """
    n = len(points)
    if n < 4:
        raise ValueError("Convex hull in 3D space requires at least 4 points")

    if isinstance(points, np.ndarray):
        points = points.tolist()
    hull_inequalities = []

    # Find the leftmost point
    leftmost = min(points, key=lambda x: x[0])

    hull = []
    p = points.index(leftmost)
    q = 0

    while True:
        hull.append(points[p])

        q = (p + 1) % n
        for i in range(n):
            if orientation_3d(points[p], points[i], points[q]) == 2:
                q = i

        p = q

        if p == 0:
            break

    # Compute plane equations for each face of the convex hull
    for i in range(len(hull) - 1):
        v1 = hull[i]
        v2 = hull[i + 1]
        normal = (v2[1] - v1[1], v1[0] - v2[0], 0)  # Cross product with Z-axis to get normal vector
        d = -(normal[0] * v1[0] + normal[1] * v1[1] + normal[2] * v1[2])
        hull_inequalities.append((normal, d))

    # Handle the last face
    v1 = hull[-1]
    v2 = hull[0]
    normal = (v2[1] - v1[1], v1[0] - v2[0], 0)
    d = -(normal[0] * v1[0] + normal[1] * v1[1] + normal[2] * v1[2])
    hull_inequalities.append((normal, d))

    return hull_inequalities

def draw_convex_hull_3d(points, hull, point = None) -> None:
    for i, p in enumerate(points):
        print(f'p: {p} - {i}')
    i = 0
    col = ['r','b','g','r','d']
    #fig, axs = plt.subplots(3,2, figsize=(8, 10),subplot_kw={'projection': '3d'})
    #axs = axs.flatten()

    for simplex in hull.simplices:
        #axs[i].scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o')
        #col_i = col[i%len(col)]
        #axs[i].plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], col_i+'-',)
        i+=1
    #plt.show()
    i = 0
    fig = plt.figure('Convex hull computation')
    
    ax = fig.add_subplot(111, projection='3d')
   #     draw_grid_3d()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o')
    if point is not None:
        ax.scatter(*point, c='k', marker='x')
    for simplex in hull.simplices:
        col_i = col[i%len(col)]
       
        triangle = Poly3DCollection([points[simplex]], facecolors='cyan', edgecolors='r', alpha=0.5)
        ax.add_collection3d(triangle)

        i+=1
    plt.show()



def is_inside_convex_hull(point, convex_hull_inequalities):
    """
    Check if a point is inside the convex hull defined by the inequalities.
    Returns:
    - True if the point is inside or on the boundary of the convex hull.
    - False otherwise.
    """
    for normal, d in convex_hull_inequalities:
        if np.dot(normal, point) + d > 0:
            return False
    return True
if __name__ == "__main__":
    # Example usage
    try_ = False
    if try_:
        num_points = 10
        min_coord, max_coord = -1, 1
        points_3d = [(0, 0, 3),(-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0) ]
        convex_hull_inequalities = jarvis_march_3d(points_3d)
        print(convex_hull_inequalities)
        random_points = np.random.uniform(min_coord, max_coord, size=(num_points, 3))
        for point_to_check in random_points:
        #point_to_check = np.array([-0.5, 0, 0.7])  # Point to check
            print("The point is: ", point_to_check)
            inside = is_inside_convex_hull(point_to_check, convex_hull_inequalities)
            if inside:
                print("Point is inside or on the boundary of the convex hull.")
            else:
                print("Point is outside the convex hull.")

            plot_convex_hull_3d(points_3d, convex_hull_inequalities, point_to_check=point_to_check)
