import numpy as np
from numpy.linalg import norm
import scipy as sp
rng = np.random.default_rng(12345)
#from numba import jit
from scipy.optimize import minimize


class Pedestrian:
    """Constructs a Pedestrian. Add target for preferred direction.
    Also fix direction change billiard style
    Parameters
    ----------
    steps : int, optional
            Number of steps the Pedestrian will perform.
    speed : float, optional
            determines the size of each step of the Pedestrian.
    
    Returns
    -------
    out :  object
    """
    
    def __init__(self, steps=20, 
                 speed_mu=1.3, 
                 target="Random",
                 position="Random", 
                 personal_space=10, 
                 obstacles=[[np.zeros(2),np.zeros(2)]],
                 others_sensitivity=5, 
                 obstacles_sensitivity=10e8,
                 obstacles_threshold=2):
        self.speed = np.abs(rng.normal(speed_mu,0.2)) # m/s
        if type(target) != np.ndarray:
            self.target = np.array([rng.uniform(0,100),rng.uniform(0,100)])
        else:
            self.target = target
        if type(position) != np.ndarray:
            self.position = np.array([rng.uniform(0,100),rng.uniform(0,100)])
        else:
            self.position = position
        self.direction = ((self.target - self.position)/norm(self.target - self.position))            
        self.direction_outlook = self.direction
        self.personal_space = personal_space
        self.history = np.zeros(2*(steps+1)).reshape(2,steps+1)
        self.current_step = 0
        self.obstacles = obstacles
        self.direction_outlook_theta = 0
        self.others_sensitivity = others_sensitivity
        self.obstacles_sensitivity = obstacles_sensitivity
        self.obstacles_threshold = obstacles_threshold
        self.steps = steps
    
                 
    def expected(self):
        return self.position + self.speed * self.direction
    
    def direction_theta(self, direction):
        """For a vector return the angle with respect to [1,0]"""
        direction = direction/norm(direction)
        theta = np.arccos(direction[0])
        if direction[1] < 0:
            theta = 2*np.pi - theta
        return theta

    
    
    def assess_players(self,other_pedestrians):
        assert type(other_pedestrians) == list, "Pass a list of pedestrians."
        self.assess_positions = np.zeros(2*len(other_pedestrians)).reshape(2,len(other_pedestrians))
        for i, ped in enumerate(other_pedestrians):
            self.assess_positions[:,i] = ped.expected() 
    
    
    def _assess_obstacles(self,exp_pos_obs):
        """If the pedestrian is expeceted to be close to an obstacle an it expects to want to move toward the 
           obstacle, then return a high loss."""
        obstacles = self.obstacles
        dist_to_obstacles = 10e7*np.ones(len(obstacles))
        for i , obstacle in enumerate(obstacles):
            x = obstacle[0] - exp_pos_obs
            y = obstacle[1] - exp_pos_obs
            z = self.target - exp_pos_obs
            cos_xy = np.round(np.dot(x,y)/(norm(x)*norm(y)),15)
            cos_xz = np.round(np.dot(x,z)/(norm(x)*norm(z)),15)
            cos_yz = np.round(np.dot(y,z)/(norm(y)*norm(z)),15)  
            angle_xy = np.arccos(cos_xy)
            angle_xz = np.arccos(cos_xz)
            angle_yz = np.arccos(cos_yz)
            if (angle_xy +.2 >= angle_xz) & (angle_xy +.2 >= angle_yz): # next direction towards obstacle
                cos_ = np.dot(obstacle[1]-obstacle[0],exp_pos_obs-obstacle[0])/(norm(obstacle[1]-obstacle[0])*norm(exp_pos_obs-obstacle[0]))
                dist = (norm(exp_pos_obs-obstacle[0])* np.sqrt(1-cos_**2))
                if dist < self.obstacles_threshold:
                    dist_to_obstacles[i] = dist
        return 1/dist_to_obstacles
            
    
        
    def walk_theta(self):
        """Finds optimal step and compromise step by minimizing the loss function."""
        # I fix current preferred direction
        preferred = ((self.target - self.position) / norm(self.target - self.position)) 
        # Loss function
        def f(theta):
            theta = theta[0]
            direction_theta = np.array([np.cos(theta), np.sin(theta)])
            exp_pos = self.position + self.speed * direction_theta
            exp_pos_obs = self.position + 2* self.speed * direction_theta # 2 --> Longer sight for assessing obstacles
            dissatisfaction_ped = norm(self.assess_positions - exp_pos.reshape(2,1), axis=0)
            dissatisfaction_ped = self.others_sensitivity/(.1 + dissatisfaction_ped)
            dissatisfaction_obs = self.obstacles_sensitivity*self._assess_obstacles(exp_pos_obs)
            inconvenience_f = norm(preferred - direction_theta)**2  # round theta to e-15 
            return inconvenience_f + np.sum(dissatisfaction_ped**2) + np.sum(dissatisfaction_obs)
        x0 = self.direction_theta(self.direction)
        # argmin on loss function
        theta_ = minimize(f,x0).x[0]
        self.direction_outlook = np.array([np.cos(theta_), np.sin(theta_)])
        self.position += self.speed * self.direction_outlook
        self.current_step += 1
        self.history[:,self.current_step] = self.position
        self.direction = self.direction_outlook
        # Stop moving if arrived at target
        if norm(self.position - self.target) < 3:
            self.speed = 0