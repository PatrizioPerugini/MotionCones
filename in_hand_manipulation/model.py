import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from utils import plot_convex_hull
import sympy as sp
from elimina import draw_convex_hull_3d
# initial_state=np.array([0.5,0,0]) ||| initial_state=np.array([0,0,0])
# goal_state=np.array([.45,0.5,0])  ||| goal_state=np.array([-0.2,0.2,0]

#in this configuration [0,0.2,0] ([0.2, 0.3, 0] too) it is able to get the goal with the rotation. My guess is that
#its working properly and the wall below allows this motion.. but we have to see if this is the case


#goal_state=np.array([0.1,0.2,-0.0]) # o trova subito
class Model:
    def __init__(self,x_dim =3, m = 1, g=9.81, 
                initial_state=np.array([0,0,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), eps_goal = 0.007, dt = 1) -> None:
        #configuration space
        self.x_dim = x_dim
        self.m = m
        self.g = g
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps_goal = eps_goal
        self.dt = dt

        self.r = 1/100#0.1*10  # cm
        self.rc_q = (0.6 * self.r) ** 2
        self.m = 500/1000
        #self.mu_s = 1 #0.5
        self.mu_s = 0.5
        self.N=40
        #self.mu = 0.5#0.7
        self.mu = 0.9
        self.theta_mu =math.atan(self.mu)

        ##object geometry in cm -> it's a square object of dim 10x10

        self.c = math.cos(self.theta_mu)
        self.s = math.sin(self.theta_mu)
        self.d = 5/100  #cm
        self.w = 10/100#cm
        #pi/2 -> pusher a dx,  0 -> pusher sotto, -pi/2 -> pusher sx
        #self.B = np.diag([1,1,1/0.6*self.r])
        self.B = np.diag([1,1,1/self.rc_q])


        #ADD THE DIFFERENT WALL ORIENTATIONS IN THE ENVIRONMENT
        self.wall_orientations = [0,-np.pi/2, np.pi/2]#-np.pi]
        #self.wall_orientations = [-np.pi/2]
        #self.wall_orientations = [np.pi/2]
        #self.wall_orientations = [np.pi]
        #self.wall_orientations = [0] #[0.2,0.3,0]-> muro sotto

    
    def compute_rigid_transform(self,x):
        #input: state (x,z,theta)
        #output: transform from world to object frame
        
        c = math.cos(x[2])
        s = math.sin(x[2])
        tx = x[0]
        tz = x[1]
        T = np.array([[c, -s, tx],[s, c, tz], [0, 0, 1]])
        return T

  
    #NB-> it should take into account all the different pushers that are available... citing the paper
    #"generate_motionCones() - get_motionCones() - computes the polyhedral motion
    # cones for a given object configuration in the grasp for all possible external pushers"
    
    #x:np.ndarray->state
    def get_motionCones(self, x, theta_rad): #somehow I should also extend it to have a greater number of pushers
        fx_h, fz_h, my_h, k = sp.symbols("fx_h fz_h my_h k")
        num_sol = 4
        J_s = self.compute_rigid_transform(x)

        
        #f_r = [self.mu,1]
        #f_l = [-self.mu,1]
        f_r = [np.cos(np.pi/2-math.atan(self.mu)),np.sin(np.pi/2-math.atan(self.mu))]
        f_l = [-np.cos(np.pi/2-math.atan(self.mu)),np.sin(np.pi/2-math.atan(self.mu))]

        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
              [np.sin(theta_rad), np.cos(theta_rad)]])

        
        R_y = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                [np.sin(theta_rad), np.cos(theta_rad), 0],
                [0, 0, 1]])
        
        #J_p1 = np.array([[self.c, self.c], [self.s, -self.s], [self.d * self.c  * self.w * self.s, self.d * self.c - 0.5 * self.w * self.s]])
        
        #J_p1 = R_y@np.array([[self.c, self.c], [self.s, -self.s], [self.d * self.c  + 0.5 *self.w * self.s, self.d * self.c - 0.5 * self.w * self.s]])
#
        #d = -self.d
        #J_p2 = R_y@np.array([[self.c, self.c], [self.s, -self.s], [d * self.c + 0.5 * self.w * self.s, d * self.c - 0.5 * self.w * self.s]])
        #J_p2 = np.array([[self.c, self.c], [self.s, -self.s], [d * self.c + 0.5 * self.w * self.s, d * self.c - 0.5 * self.w * self.s]])

        J_p1 = np.array([[1, 0], 
                     [0, 1], 
                     [-self.w/2, self.d]])
        #d = -d
        J_p2 = np.array([[1, 0], [0, 1], [-self.w/2, -self.d ]])


        #wp1_r = np.dot(J_p1, f1_r)
        #wp1_l = np.dot(J_p1, f1_l)
        #wp2_r = np.dot(J_p2, f2_r)
        #wp2_l = np.dot(J_p2, f2_l)

        wp1_r = np.dot(J_p1, f_r)
        wp1_l = np.dot(J_p1, f_l)
        wp2_r = np.dot(J_p2, f_r)
        wp2_l = np.dot(J_p2, f_l)


       

        #J_p1 = J_p1
        #J_p2 = J_p2
        norm = 1#self.N*self.mu,
        wrench_cone = np.vstack([wp1_r, wp1_l, wp2_r, wp2_l])
        sol = []
        for edge_wrench in wrench_cone:
            w_pusher_hat=sp.Matrix([edge_wrench[0]/norm, edge_wrench[1]/norm, edge_wrench[2]/norm])
            fx_h_vec = sp.Matrix([fx_h, fz_h, my_h])#w_s_hat                                                           # sp.Matrix([0, 0, -9.8])
                          #J_s
            stab_push_eq = J_s.T * fx_h_vec - k/(-self.mu_s*self.N) * w_pusher_hat - self.m/(-self.mu_s*self.N) * sp.Matrix([0, 9.8, 0 ])
   
            eq_ellips = sp.Eq(fx_h**2/1 + fz_h**2/1 + my_h**2/self.rc_q, 1)

            solution = sp.solve([eq_ellips, stab_push_eq], [fx_h, fz_h, my_h, k])
            #print(solution)
            if(solution[0][num_sol-1]> 0):
                sol.append(solution[0])
            else:
                sol.append(solution[1])

        solutions = np.array(sol)
        #w_s_hat = np.array([[solutions[0][0:3]],[solutions[1][0:3]],[solutions[2][0:3]],[solutions[3][0:3]]])

        ws_h = []
        for i in range(len(sol)):
            ws_h.append(solutions[i][0:num_sol-1])
        v_obj = []
        for ws in ws_h:
            ws = np.reshape(ws,(3,1))
            #v_i = np.linalg.inv(J_s)@self.B@(ws)
            v_i = J_s@self.B@(ws)
            v_obj.append(v_i)

        V_obj = np.reshape(np.array(v_obj,dtype = float), (num_sol,3,))  
        #x must be added bc is the reference frame for this convex hull
        
        #this is right for four contact
        V_origin =np.array([0,0,0])
        #check_hull_vel = np.vstack([V_obj[0][:],V_obj[1][:],V_obj[2],V_obj[3],V_origin])
        check_hull_vel = np.vstack([V_obj[0][:],V_obj[1][:],V_obj[2],V_obj[3]])
        #check_hull_vel = np.vstack([V_obj[0][:],V_obj[1][:],x])
        #print("Hull equations are: ", check_hull_vel)
        hull = ConvexHull(check_hull_vel)
        
        #print(hull.equations)
        #plot_convex_hull(check_hull_vel,hull)
        points = np.array([V_obj[0],V_obj[1],V_obj[2],V_obj[3]])
        draw_convex_hull_3d(check_hull_vel,hull)
       
        #print(check_hull_vel[hull.vertices])

        return hull

    def goal_check(self, x):
        min_dist = np.inf
        x_ = x.copy()
        goal = False
        #for goal_state in self.goal_states:
        #dist = np.linalg.norm(x_-self.goal_state)
        dist_xz = np.linalg.norm(x_[0:2]-self.goal_state[0:2])
        dist_y = np.linalg.norm(x_[2]-self.goal_state[2])
        dist = dist_xz + dist_y*0.01
        if dist<min_dist:
            min_dist = dist

        if min_dist < self.eps_goal:
            goal = True
        
        return goal, min_dist
        
    

    def sample(self):
        goal_bias = np.random.rand(1)
        #if goal_bias < 0.3:
        #    
        if goal_bias < 0.4: return self.goal_state
        #    else:                return self.goal_state # this might have sense if we consider the orientation to be the same every pi, but don't think is
        #                                                    # useful
        #else:
        rnd = (np.random.rand(3)-0.5)*2# range between -1 and 1
        #rnd[0]*= 0.4
        #rnd[1]*= 0.2

        rnd[2]= (np.random.rand(1) - 0.5) * 0.7 * np.pi/4
        #print(rnd)
        #print("this is the input from the function sample in model ")
        #input()
        return rnd