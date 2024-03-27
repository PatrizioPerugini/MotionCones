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

np.set_printoptions(precision=4,linewidth=np.inf, suppress=True)

class MotionPlanner:
    def __init__(self, N_steps, m = 500/1000, g=9.81, 
                initial_state=np.array([0,0,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), wall_orientations = [0, np.pi/2, -np.pi/2],  r = 1/100, mu_s = 0.9,
                N_gripper = 50, mu = 0.9, d = 5/100, w = 10/100 ):
        # Initialize parameters
        self.m = m
        self.g = g
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.callback_iteration = 0
        self.obj_reached = False
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


        self.MC_solver = MotionCone()
        self.MC_solver.get_functions()

        #in theory this should give me the possibility to include multiple pushers
 
        self.wall_orientations = wall_orientations
        self.N_pushers = len(self.wall_orientations)

        #self.wall_orientations = [-np.pi,np.pi/2]
        
        self.initialize_parameters()
    
    def initialize_parameters(self):

        self.a = -1/(self.mu_s*self.N_gripper)
        self.b = self.m*self.g*self.a

        self.B = np.diag([1,1,1/self.rc_q])
        self.J_p1 = np.array([[1, 0], [0, 1], [-self.w/2, self.d]])
        self.J_p2 = np.array([[1, 0], [0, 1], [-self.w/2, -self.d]])

        self.f_r = [np.cos(np.pi/2-math.atan(self.mu)), np.sin(np.pi/2-math.atan(self.mu))]
        self.f_l = [-np.cos(np.pi/2-math.atan(self.mu)), np.sin(np.pi/2-math.atan(self.mu))]
        
        #self.f_r = [self.mu,1]
        #self.f_l = [-self.mu,1] 

    def apply_jacobian_rotation(self, theta_rad):

        R_y = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                        [np.sin(theta_rad), np.cos(theta_rad), 0],
                        [0, 0, 1]])
        wp1_r = np.dot(R_y@self.J_p1, self.f_r)
        wp1_l = np.dot(R_y@self.J_p1, self.f_l)
        wp2_r = np.dot(R_y@self.J_p2, self.f_r)
        wp2_l = np.dot(R_y@self.J_p2, self.f_l)
        return wp1_r, wp1_l, wp2_r, wp2_l
        
    def compute_rigid_transform(self, x):
        # Compute rigid transform from world to object frame
        c = math.cos(x[2])
        s = math.sin(x[2])
        tx = x[0]#0
        tz = x[1]#0
        T = np.array([[c, -s, tx], [s, c, tz], [0, 0, 1]])
        return T

    def get_motionCones(self, x, theta_rad, vel_i=None):
        num_sol = 4
        J_s = self.compute_rigid_transform(x)

       
        wp1_r, wp1_l, wp2_r, wp2_l = self.apply_jacobian_rotation(theta_rad)

        wrench_cone = np.vstack([wp1_r, wp1_l, wp2_r, wp2_l])

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
            v_obj.append(v_i)
        V_obj = np.reshape(np.array(v_obj,dtype=float), (num_sol, 3,))
        V_origin = np.array([0, 0, 0])

        check_hull_vel = np.vstack([V_obj[0][:], V_obj[1][:], V_obj[2], V_obj[3], V_origin])
        if np.isnan(check_hull_vel).any():
            print("UUUPS")
            return None, None
        else:
            hull = ConvexHull(check_hull_vel)
            return hull, check_hull_vel


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


    def compute_positions_probabilities(self, velocities_x, velocities_z, velocities_w, T,c):
        dt = T/ (self.N_steps-1)
        p_n_x, p_n_z, p_n_w = self.initial_state
        x_positions = [p_n_x]
        z_positions = [p_n_z]
        w_positions = [p_n_w]

        for t in range(self.N_steps):
            vx, vz, vw = 0, 0, 0
            #sume over all combinations velocity pusher
            for j in range(self.N_pushers):
                vx += velocities_x[t*self.N_pushers+j]*c[t*self.N_pushers+j]
                vz += velocities_z[t*self.N_pushers+j]*c[t*self.N_pushers+j]
                vw += velocities_w[t*self.N_pushers+j]*c[t*self.N_pushers+j]
            update = box_plus(p_n_x, p_n_z, p_n_w, vx * dt, vz * dt, vw * dt)
            p_n_x, p_n_z, p_n_w = update.flatten()
            x_positions.append(p_n_x)
            z_positions.append(p_n_z)
            w_positions.append(p_n_w)

        return x_positions, z_positions, w_positions

    def kl_divergence(self, c):
        kl = []
        for i in range(0,len(c)-self.N_pushers,self.N_pushers):
            p = c[i:i+self.N_pushers]
            q = c[i+self.N_pushers:i+2*self.N_pushers]    
            kl_i = 0     
            for p_i, q_i in zip(p,q): 
                if p_i == 0 or q_i == 0:
                    kl_i += 0
                else:
                    kl_i += p_i*np.log(p_i/q_i)
            kl.append(kl_i)
        return kl


    def objective_function(self, vars):
        T = vars[0]  # Total time
        #vx for each pusher and for each timestep
        velocities_x = vars[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        velocities_z = vars[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        angular_velocities_w = vars[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = vars[3*self.N_pushers*self.N_steps+1:]
        

        p_x, p_z, p_w = self.compute_positions_probabilities(velocities_x, velocities_z, angular_velocities_w, T,c)
        p_n_x, p_n_z, p_n_w = p_x[-1], p_z[-1], p_w[-1]
        dist_xz = np.linalg.norm(np.array([p_n_x, p_n_z]) - self.goal_state[0:2])
        dist_w = np.linalg.norm(np.array([p_n_w]) - self.goal_state[2])
        dist = 0.2*dist_w + 0.8*dist_xz

        entropy_cost = sum(c[t] * np.log(c[t]) for t in range(self.N_steps * self.N_pushers) if c[t] != 0)

        lambda_e = 0.01#0.005
        kl_term = np.sum(self.kl_divergence(c))
        lamda_kl = 0.00001
        return   (1-lambda_e)*dist - lambda_e*entropy_cost #- lamda_kl*kl_term

    def eq_constraints(self, vars):
        
        c = vars[3*self.N_pushers*self.N_steps+1:]
        constraints = []
        c_m = 0
        for n in range(self.N_steps*self.N_pushers):
            if n%self.N_pushers==0 and n!=0:
                constraints.append(c_m -1)
                c_m = 0
            c_m+=c[n]
            if n==self.N_steps*self.N_pushers -1:
                constraints.append(c_m -1)

        return np.array(constraints)
    
    def compute_hp(self,a, b, v):
        #ax+b
        dp = np.dot(a, v) - b #- b
        return dp
    
    def poc_ch(self,theta):
        R_y = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        points_before_rotation = np.array([[.01, 0, .01],
                                   [.01, -.01, .01],
                                   [-.01, .01, 0],
                                   [-.01, -.01, -.01]])
                        
        points_after_rotation = np.dot(points_before_rotation, R_y.T)

        convex_hull = ConvexHull(points_after_rotation)

        return convex_hull
    def get_square_const(self,x, z, inequalities):
            inequalities.extend([-self.d/10+x])
            inequalities.extend([-self.d/10+z])
            inequalities.extend([-self.d/10-x])
            inequalities.extend([-self.d/10-z])
            return inequalities
    
    #center it where needed -> need to be generalized
    def sigmoid(self,x):
        x_ = x%.1
        return 1 / (1 + np.exp(-300 * (x + 0.02)))

    #TODO compute the constraints in a class object, to extend to multiple objects shape
    def get_t_constraints(self, x, z, constraints):

        c = self.sigmoid(x)
        epsilon = 0.003
        a = 0.015  - epsilon
        b = 0.05   - epsilon
  
        #using the soft constraint
        constraints.extend([c*(z+a)*(-z+a)+(1-c)*(z+b)*(-z+b)])
        constraints.extend([x + 0.05])
        constraints.extend([-x + 0.05])        


        return constraints

    def ineq_constraints(self, vars):
        T = vars[0]  # Total time
        velocities_x = vars[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        velocities_z = vars[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        velocities_w = vars[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = vars[3*self.N_pushers*self.N_steps+1:]
 
        inequalities = []
        twists = []
        for vx, vz, vw in zip(velocities_x,velocities_z,velocities_w):
            current_twist = [vx, vz, vw]
            twists.append(current_twist)
        for _ in range(self.N_pushers):
            twists.append([0, 0, 0])
        
        p_x, p_z, p_w = self.compute_positions_probabilities(velocities_x, velocities_z, velocities_w, T, c)

        for i, ( x, z, w)  in enumerate(zip(p_x, p_z, p_w)):#, velocities_x, velocities_z, velocities_w):
            current_state = np.array([x, z, w])

            # SQUARE INEQUALITIES
            #inequalities = self.get_square_const(inequalities)
       
            # T-SHAPE INEQUALITIES -> dim of t are: 4 and 10 for bases 3 and 7 for height (incavata in 10x10 square)
            #inequalities = self.get_square_const(x, z, inequalities)
            inequalities = self.get_t_constraints(x, z, inequalities)
            
          
        # Compute motion cones for the given state and orientations of the wall. for now zero
            for j,theta_rad in enumerate(self.wall_orientations):
                #'''
                motion_cones_i, _ = self.get_motionCones(current_state, theta_rad = theta_rad)  
                if motion_cones_i == None:
                    for _ in range(6):
                        inequalities.append(0)
                    continue
                    motion_cones_i = [0,0,0]
                #'''

                #motion_cones_i  = self.poc_ch(theta_rad)#ConvexHull(np.array([[1,0,1],[1,-1,1],[-1,1,0],[-1,-1,-1]]))
                
                #idx = print((i-1)*(self.N_pushers)+j)
                #print((i)*(self.N_pushers)+j)

                current_twist = np.array(twists[(i)*(self.N_pushers)+j])
                #print("current twist", current_twist)
                #cnt = 0
                for equation in motion_cones_i.equations:

                    a = equation[:-1]  # Coefficients a1, a2, ..., an
                    b = equation[-1]   # Constant term b

                    # Constructing inequality: a1*x1 + a2*x2 + ... + an*xn + b <= 0
                    #inequality = {"type": "ineq", "fun": lambda x, a=a, b=b: np.dot(a, x) - b}
                    inequality = self.compute_hp(a, b, current_twist)
                    #print("inequality is: ", inequality)
                    inequalities.append(inequality)
                    #cnt+=1
                #print("cnt: ", cnt)
                #input("ops")
        c_matrix = np.array(c).reshape((self.N_steps, self.N_pushers))
    
        flattened_c = c_matrix.flatten()
    
        for c_ij in flattened_c:
            inequalities.append(c_ij)  

        return inequalities

    
    def retrieve_optim_vels(self,v_x,v_z,v_w,pushers):
        vx = []
        vz = []
        vw = []
        for t in range(self.N_steps):
            best_pusher = -1
            idx = 0
            best_idx = 0
            for j in range(self.N_pushers):
                if pushers[t*self.N_pushers+j] > best_pusher:
                    best_pusher = pushers[t*self.N_pushers+j]
                    best_idx = idx
                idx+=1
            if t*self.N_pushers+best_idx < len(v_w):
                v_x_t = v_x[t*self.N_pushers+best_idx]
                v_z_t = v_z[t*self.N_pushers+best_idx]
                v_w_t = v_w[t*self.N_pushers+best_idx]
                vx.append(v_x_t)
                vz.append(v_z_t)
                vw.append(v_w_t) 
        return vx, vz, vw

    def callback_func(self, xk):
        T = xk[0]
        dt = T/(self.N_steps-1)
        v_x = xk[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        v_z = xk[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        v_w = xk[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = xk[3*self.N_pushers*self.N_steps+1:]


        #get list of positions
        vx, vz, vw = self.retrieve_optim_vels(v_x,v_z,v_w,c)

        v_i = [[vx_,vz_,vw_] for vx_,vz_,vw_  in zip(vx,vz,vw)]

        #px,py,pz = self.compute_positions(v_x, v_z, v_w, self.Traj_time)
        px,py,pz = self.compute_positions_probabilities(v_x, v_z, v_w, T,c)
        curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
        print(f"\nIter. {self.callback_iteration}: OF = {self.objective_function(xk)}  -  dt = {T/(self.N_steps-1)}")
        print("{: <10} {: <30} {: <30}".format('TIMESTEP', 'POSITIONS', 'VELOCITIES'))
        t = 0
        for v,p in zip(v_i,curr_state):
            print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in p]), str([round(i,4) for i in v])))
            t += 1
        print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in curr_state[-1]]), str([0., 0., 0.])))
        print(c)

        if self.objective_function(xk) < 1e-3:
            self.obj_reached = True

        self.callback_iteration += 1


    def vels_ig(self, T_ig):
        delta_v = (self.goal_state-self.initial_state)/T_ig
        ddv = delta_v#/(self.N_pushers-1)
        #print(ddv)
        
        return ddv



    def get_ig(self):
        T_ig = 0.1
        ddv = self.vels_ig(T_ig)
        initial_guess = [T_ig] + [ddv[0]]*self.N_steps*self.N_pushers + [ddv[1]]*self.N_steps*self.N_pushers + [ddv[2]]*self.N_steps*self.N_pushers +  [1/self.N_pushers]*self.N_steps*self.N_pushers
        #initial_guess = [T_ig] + [-0.] *3*self.N_steps*self.N_pushers + [1/self.N_pushers]*self.N_steps*self.N_pushers
        return  initial_guess



    #velocities = [v_x_0, v_x_1, v]
    def optimize_motion_cone(self):
        
        #bouns on: T + 3NM + NM
        bounds = [(0.05, 30)] + [(-2, 2)] * (3*self.N_pushers*self.N_steps) + [(0, 1)] * (self.N_steps*self.N_pushers)  # Bounds for T and velocities

        initial_guess = self.get_ig()

        constraints = [{'type': 'eq', 'fun': self.eq_constraints},
                   {'type': 'ineq', 'fun': self.ineq_constraints}]
        #options = {'maxiter': 190, 'disp': True, 'ftol':1e-4, 'gtol':1e-1, 'maxfev' :5000}
        options = {'maxiter': 200, 'disp': True, 'ftol':1e-7}#1e-7
        result = minimize(self.objective_function, initial_guess, constraints=constraints, bounds=bounds, options=options, 
                            callback=self.callback_func, method='SLSQP')
        
                                                                #trust-constr
        if self.obj_reached or result.success: #result.success:
            T = result.x[0]
            velocities = result.x[1:3*self.N_pushers*self.N_steps+1]
            pushers =  result.x[3*self.N_pushers*self.N_steps+1:]
            v_x = result.x[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
            v_z = result.x[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
            v_w = result.x[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]
            print("Optimal Time T:",T)
            print("velocities:", velocities )
            print("pushers:", pushers)
            vx, vz, vw = self.retrieve_optim_vels(v_x,v_z,v_w,pushers)
            
            # = []
            # = []
            # = []
            #for t in range(self.N_steps):
            #    best_pusher = -1
            #    idx = 0
            #    best_idx = 0
            #    for j in range(self.N_pushers):
            #        if pushers[t*self.N_pushers+j] > best_pusher:
            #            best_pusher = pushers[t*self.N_pushers+j]
            #            best_idx = idx
            #        idx+=1
            #    if t*self.N_pushers+best_idx < len(v_w):
            #        v_x_t = v_x[t*self.N_pushers+best_idx]
            #        v_z_t = v_z[t*self.N_pushers+best_idx]
            #        v_w_t = v_w[t*self.N_pushers+best_idx]
#
            #        v_x.append(v_x_t)
            #        v_z.append(v_z_t)
            #        v_x.append(v_w_t)    
            #
            #v_x = result.x[1:self.N_steps+1]  # Gripper velocities for x
            #v_z = result.x[self.N_steps+1:2*self.N_steps+1]  # Gripper velocities for z
            #v_w = result.x[2*self.N_steps+1:] 
            sampled_vels =  [[vx_,vz_,vw_] for vx_,vz_,vw_ in zip(vx,vz,vw) ] 
            px,py,pz = self.compute_positions(vx, vz, vw, T) #here is correct to use it
            curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
            return T, sampled_vels, curr_state
        else:
            print("Optimization failed.")
            return None, None, None

