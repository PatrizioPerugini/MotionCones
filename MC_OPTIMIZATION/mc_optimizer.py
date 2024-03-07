import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import sympy as sp
import math
from scipy.optimize import minimize
from utils import *
import cProfile
import random

from fast_mc import MotionCone



class MotionPlanner:
    def __init__(self, N_steps, m = 500/1000, g=9.81, 
                initial_state=np.array([0,0,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), eps_goal = 0.007, dt = 1, r = 1/100, mu_s = 0.9,
                N_gripper = 100, mu = 0.9, d = 5/100, w = 10/100 ):
        # Initialize parameters
        self.m = m
        self.g = g
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps_goal = eps_goal
        self.dt = dt
        self.callback_iteration = 0

        self.r = r
        self.rc_q = (0.6 * self.r) ** 2
        self.m = m
        self.mu_s = mu_s
        self.N_gripper= N_gripper
        self.mu = mu
        self.theta_mu =math.atan(self.mu)

        self.N_steps = N_steps #number of  steps
        self.cnt=0

        self.c = math.cos(self.theta_mu)
        self.s = math.sin(self.theta_mu)
        self.d = d  
        self.w = w

        self.Traj_time = 0.1

        #instantiate the equation and load the functions for fast computation. 
        #Give a specific path to get_functions() if needed

        self.MC_solver = MotionCone()
        self.MC_solver.get_functions()
        #self.MC_bad = MotionConeBad()

        self.a = -1/(self.mu_s*self.N_gripper)
        self.b = self.m*self.g*self.a

        self.B = np.diag([1,1,1/self.rc_q])
        self.J_p1 = np.array([[1, 0], [0, 1], [-self.w/2, self.d]])
        self.J_p2 = np.array([[1, 0], [0, 1], [-self.w/2, -self.d]])

        self.f_r = [np.cos(np.pi/2-math.atan(self.mu)), np.sin(np.pi/2-math.atan(self.mu))]
        self.f_l = [-np.cos(np.pi/2-math.atan(self.mu)), np.sin(np.pi/2-math.atan(self.mu))]

        self.wp1_r = np.dot(self.J_p1, self.f_r)
        self.wp1_l = np.dot(self.J_p1, self.f_l)
        self.wp2_r = np.dot(self.J_p2, self.f_r)
        self.wp2_l = np.dot(self.J_p2, self.f_l)
        
    def compute_rigid_transform(self, x):
        # Compute rigid transform from world to object frame
        c = math.cos(x[2])
        s = math.sin(x[2])
        tx = x[0]#0
        tz = x[1]#0
        T = np.array([[c, -s, tx], [s, c, tz], [0, 0, 1]])
        return T

    def get_motionCones(self, x, theta_rad, vel_i):
        num_sol = 4
        J_s = self.compute_rigid_transform(x)

       
        
        R_y = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                        [np.sin(theta_rad), np.cos(theta_rad), 0],
                        [0, 0, 1]])
       
        wrench_cone = np.vstack([self.wp1_r, self.wp1_l, self.wp2_r, self.wp2_l])

        sol = []
        for edge_wrench in wrench_cone:

            c   = math.cos(x[2])#1
            s   = math.cos(x[2])#1
            t_x = x[0]#1
            t_z = x[1]#1
            a1 = edge_wrench[0]*self.a
            a2 = edge_wrench[1]*self.a
            a3 = edge_wrench[2]*self.a

            F_x_val, F_z_val, M_y_val, K_val = self.MC_solver.retrieve_solution(c,s,t_x,t_z,a1,a2,a3,self.b,self.rc_q)

            solution = np.array([F_x_val, F_z_val, M_y_val, K_val])
            sol.append(solution)
        solutions = np.array(sol)
        
        ws_h = []
        for i in range(len(sol)):
            ws_h.append(solutions[i][0:num_sol-1])
        v_obj = []
        for ws in ws_h:
            ws = np.reshape(ws, (3, 1))
            v_i = np.linalg.inv(J_s) @ self.B @ (ws)
            #v_i = self.B @ (ws)
            v_obj.append(v_i)
        V_obj = np.reshape(np.array(v_obj,dtype=float), (num_sol, 3,))
        #V_origin = vel_i#np.array([0, 0, 0])
        V_origin = np.array([0, 0, 0])

        check_hull_vel = np.vstack([V_obj[0][:], V_obj[1][:], V_obj[2], V_obj[3], V_origin])
        #check_hull_vel = np.vstack([V_obj[0][:], V_obj[1][:], V_obj[2], V_obj[3]])
        #hull = jarvis_march_3d(check_hull_vel)
        hull = ConvexHull(check_hull_vel)
        return hull, check_hull_vel


    def plot_trajectory(self, N, T, start_position, goal_position, velocities):
        delta_t = T / (self.N_steps-1)  # Time step
        positions_array_x = np.zeros(self.N_steps)  # Initialize x positions
        positions_array_z = np.zeros(self.N_steps)  # Initialize z positions
        positions_array_x[0], positions_array_z[0], _ = start_position  # Initial x and z positions
        print(velocities)
        for i in range(0, self.N_steps):  
            positions_array_x[i] = positions_array_x[i-1] + velocities[i] * delta_t

            positions_array_z[i] = positions_array_z[i-1] + velocities[N+i] * delta_t

        # Calculate the final error
        final_position = np.array([positions_array_x[-1], positions_array_z[-1], goal_position[2]])
        error = np.linalg.norm(final_position - goal_position)

        print("At optimum, the error is: ", error)

        plt.figure(figsize=(10, 6))
        plt.plot(positions_array_x, positions_array_z, marker='o', label='Gripper Position')  
        plt.scatter(goal_position[0], goal_position[1], color='r', label='Goal Position')
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.title('Gripper Trajectory (X-Z Plane)')
        plt.legend()
        plt.grid(True)
        plt.show()



    def compute_positions(self, velocities_x, velocities_z, angular_velocities_w, T):
        
        dt = T/ (self.N_steps-1)
        p_n_x, p_n_z, p_n_w = self.initial_state
        x_positions = [p_n_x]
        z_positions = [p_n_z]
        w_positions = [p_n_w]
        for vx, vz, vw in zip(velocities_x, velocities_z, angular_velocities_w):
   
            update = box_plus(p_n_x, p_n_z, p_n_w, vx * dt, vz * dt, vw * dt)
            p_n_x, p_n_z, p_n_w = update.flatten()
            x_positions.append(p_n_x)
            z_positions.append(p_n_z)
            w_positions.append(p_n_w)

        return x_positions, z_positions, w_positions




    def objective_function(self, vars):
        T = vars[0]  # Total time
        velocities_x = vars[1:self.N_steps+1]  # Gripper velocities for x
        velocities_z = vars[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities for z
        angular_velocities_w = vars[2*self.N_steps+1:]  # Angular velocities for w
        

        p_x, p_z, p_w = self.compute_positions(velocities_x, velocities_z, angular_velocities_w, T)
        p_n_x, p_n_z, p_n_w = p_x[-1], p_z[-1], p_w[-1]
        dist_xz = np.linalg.norm(np.array([p_n_x, p_n_z]) - self.goal_state[0:2])
        dist_w = np.linalg.norm(np.array([p_n_w]) - self.goal_state[2])
        dist = 0.2*dist_w + 0.8*dist_xz

        return   dist 

    def eq_constraints(self, vars):
        T = vars[0]  # Total time
        velocities_x = vars[1:self.N_steps+1]  # Gripper velocities for x
        velocities_z = vars[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities for z
        angular_velocities_w = vars[2*self.N_steps+1:]  # Angular velocities for w

        constraints = []
        positions = np.zeros((self.N_steps+1, 3))
        delta_t = T / (self.N_steps-1)

        p_n_x, p_n_z, p_n_w = self.initial_state
        i = 0
        for vx, vz, vw in zip(velocities_x, velocities_z, angular_velocities_w):

            update = box_plus(p_n_x, p_n_z, p_n_w, vx * delta_t, vz * delta_t, vw * delta_t)
            p_n_x, p_n_z, p_n_w = update.flatten()
            #positions[i+1] = np.array([p_i_x, p_i_z, p_i_w])
            positions[i+1] = np.array([ p_n_x, p_n_z, p_n_w])
            i+=1
        #constraints.extend([np.linalg.norm(positions[-1]- self.goal_state)])
        #constraints.extend([velocities_x[-1], velocities_z[-1], angular_velocities_w[-1]])
        return np.array(constraints)
    
    def cumput_hp(self,a, b, v):
        dp = np.dot(a, v) - b
        return dp
    


    
    def ineq_constraints(self, vars):
        T = vars[0]  # Total time
        velocities_x = vars[1:self.N_steps+1]  # Gripper velocities in x direction
        velocities_z = vars[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities in z direction
        velocities_w = vars[2*self.N_steps+1:]  # Angular velocities
 
        inequalities = []
        p_x, p_z, p_w = self.compute_positions(velocities_x, velocities_z, velocities_w, T)
        for x, z, w, v_x, v_z, v_w in zip(p_x, p_z, p_w, velocities_x, velocities_z, velocities_w):
            current_state = np.array([x, z, w])
            current_twist = np.array([v_x, v_z, v_w])
        # Compute motion cones for the given state and orientations of the wall. for now zero
        
            motion_cones_i, _ = self.get_motionCones(current_state, theta_rad=0, vel_i= np.array([v_x,v_z,v_w]))  # Implement this function
            #motion_cones_i  = ConvexHull(np.array([[1,0,1],[1,-1,1],[-1,1,0+self.cnt],[-1,-1,-1]]))
            for equation in motion_cones_i.equations:
                a = equation[:-1]  # Coefficients a1, a2, ..., an
                b = equation[-1]   # Constant term b

                # Constructing inequality: a1*x1 + a2*x2 + ... + an*xn + b <= 0
                #inequality = {"type": "ineq", "fun": lambda x, a=a, b=b: np.dot(a, x) - b}
                inequality = self.cumput_hp(a, b, current_twist)

                inequalities.append(inequality)

      
        return inequalities

    def callback_func(self, xk):
        T = xk[0]
        dt = T/(self.N_steps-1)
        v_x = xk[1:self.N_steps+1]  # Gripper velocities for x
        v_z = xk[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities for z
        v_w = xk[2*self.N_steps+1:] 


        #get list of positions

        v_i = [[v_x[i],v_z[i],v_w[i]] for i in range(self.N_steps)]

        #px,py,pz = self.compute_positions(v_x, v_z, v_w, self.Traj_time)
        px,py,pz = self.compute_positions(v_x, v_z, v_w, T)
        curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
        print(f"\nIter. {self.callback_iteration}: OF = {self.objective_function(xk)}  -  dt = {T/(self.N_steps-1)}")
        print("{: <10} {: <30} {: <30}".format('TIMESTEP', 'POSITIONS', 'VELOCITIES'))
        t = 0
        for v,p in zip(v_i,curr_state):
            print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in p]), str([round(i,4) for i in v])))
            t += 1
        print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in curr_state[-1]]), str([0., 0., 0.])))

        self.callback_iteration += 1


    def get_ig(self):
        T_ig = 0.1
        initial_guess = [T_ig] + [-0.] *self.N_steps + [-0.0]*self.N_steps + [0]*self.N_steps
        return  initial_guess


    def optimize_motion_cone(self):
        
        bounds = [(0.05, 30)] + [(-2, 2)] * (2*self.N_steps) + [(-5, 5)] * (self.N_steps)  # Bounds for T and velocities

        initial_guess = self.get_ig()

        constraints = [{'type': 'eq', 'fun': self.eq_constraints},
                   {'type': 'ineq', 'fun': self.ineq_constraints}]
        options = {'maxiter': 1000, 'disp': True, 'ftol': 1e-7}
        result = minimize(self.objective_function, initial_guess, constraints=constraints, bounds=bounds, options=options, 
                            callback=self.callback_func, method='SLSQP')
        if result.success:
            print("Optimal Time T:", result.x[0])
            print("Optimal Velocities:", result.x[1:])
            T = result.x[0]
            v_x = result.x[1:self.N_steps+1]  # Gripper velocities for x
            v_z = result.x[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities for z
            v_w = result.x[2*self.N_steps+1:] 

            
            px,py,pz = self.compute_positions(v_x, v_z, v_w, T)
            curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
            return T, result.x[1:], curr_state
        else:
            print("Optimization failed.")
            return None, None, None

