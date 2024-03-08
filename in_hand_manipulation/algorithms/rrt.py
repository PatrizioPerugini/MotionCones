from algorithms.planner import Planner, Node
from model import Model
from utils import point_in_hull, plot_hull_and_points, project_onto_convex_hull, find_closest_projection,transition_test, plot_convex_hull
import numpy as np
from elimina import draw_convex_hull_3d
class RRT(Planner):
    def __init__(self, model: Model, thr = 1e-5) :
        super().__init__(model, thr)
        
    #each node has also the information needed to retrieve in which reachable set of the parent it was.
    
    def add_node(self, state, parent:Node = None, cost = None, dt = None, from_rs = None):

    
        wall_orientations = self.model.wall_orientations
        reachable_motion_cones = []
        for theta_wall in wall_orientations:
            offset = 0#np.pi/2
            reachable_twist = self.model.get_motionCones(state,theta_wall+offset)
            reachable_motion_cones.append(reachable_twist)
        
        
        node = Node(state,parent, cost, dt, from_rs, reachable_motion_cones)
        if parent is not None:
            
            is_new = parent.add_child(node)
            if not is_new:
                return None
        self.n_nodes+=1
        state_id = self.state_tree.insert(state)
        self.id_to_node[state_id] = node


        return node
    
    def take_unit_step(self, x_parent, x_rand, step_size):
        # Calculate direction vector
        direction = np.array([x_rand[0] - x_parent[0], x_rand[1] - x_parent[1], x_rand[2] - x_parent[2]])

        # Normalize direction vector
        norm_direction = np.linalg.norm(direction)
        unit_direction = direction / norm_direction if norm_direction != 0 else direction

        # Determine step size and take step
        x_sample = np.array([
            x_parent[0] + step_size * unit_direction[0],
            x_parent[1] + step_size * unit_direction[1],
            x_parent[2] + 0.3*step_size * unit_direction[2]
        ])

        return x_sample

    #expand will now return: node_next, node_near, reach_wall
                            #step_size = 0.3
    def expand(self, x_rand, step_size = 0.3):
        
        id_near = self.state_tree.nearest(x_rand)
        node_near = self.id_to_node[id_near]

        x_near = node_near.state
        
        #move from x_near taking a unit step in the direction of q_rand
        x_sample = self.take_unit_step(x_near, x_rand, step_size)

        #check if the object twist to get there is within the reachable object twist of this parent node.
        #transition_test
        if not transition_test(x_near,x_sample,self.model.goal_state):
            return None, None

        sampled_twist = (x_sample-x_near)/self.model.dt
        
  

        reach_wall = -1
        for reachable in node_near.reachable:
            
            is_reachable = point_in_hull(sampled_twist,reachable, tolerance=1e-12)
            #if reach_wall is zero the first pusher is used and so on
            if is_reachable:
                reach_wall+=1
                break
        #plot_hull_and_point(node_near.reachable, sampled_twist)
        #project_onto_convex_hull(node_near.reachable, sampled_twist)
        
        grasp_maintained = True # check the position of the ee
       
        #if already in the motion cone then add it directly
        if(is_reachable and grasp_maintained ):
            x_next = x_sample
            cost = 0.001
            node_next = self.add_node(x_next, node_near, cost, self.model.dt,reach_wall)
           
            return node_next, node_near#, reach_wall

        #else project the object twist into the convex hull
        else:
            #keep track of the wall
            i = -1
            closest_ = np.inf
            reach_wall = -1
            closest_proj_twist = None

           

            for reachable in node_near.reachable:

                proj_twist, distance = find_closest_projection(reachable, sampled_twist)
                #plot_hull_and_points(reachable, sampled_twist, proj_twist)

                i+=1

                
                if distance < closest_:
                    closest_ = distance
                    reach_wall = i
                    closest_proj_twist = proj_twist

 
 
            x_next =x_near + self.model.dt*closest_proj_twist
            if not transition_test(x_near,x_next,self.model.goal_state):
                #print("NOT GOOD PROJECTION ")
                return None, None

            cost = 0.001

            node_next = self.add_node(x_next, node_near, cost, self.model.dt, reach_wall)
            #print("node next is: ", node_next)
            return node_next, node_near
   



        