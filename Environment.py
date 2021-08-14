import numpy as np
from numpy.linalg import norm

rng = np.random.default_rng(12345)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()       

class Environment:
    """Defines an evironment by feeding it a list  segments, defined by two 
       endoints each."""
    
    def __init__(self, obstacles, name="N/A"):
        self.obstacles = obstacles
        self.name = name
           
       
    def obstacles_to_plot(self, size=150):
        obstacles_plot = []
        for obstacle in self.obstacles:
            obs_dir = obstacle[1]-obstacle[0]
            L = np.linspace(0,norm(obs_dir),size)
            barrier = np.zeros(2*size).reshape(2,size)
            for i, l in enumerate(L):
                    barrier[0,i] = obstacle[0][0] + l*(np.dot(obs_dir,np.array([1,0]))/norm(obs_dir))
                    barrier[1,i] = obstacle[0][1] - l*np.cross(obs_dir,np.array([1,0]))/norm(obs_dir)
            obstacles_plot.append(barrier)
        return obstacles_plot 
    
    
    def plot(self):
        plt.xlim(-20,100)
        plt.ylim(-20,100)
        plt.title(f"{self.name}")
        for obstacle_ in self.obstacles_to_plot():
            plt.scatter(obstacle_[0],obstacle_[1], s=4, label=f"Barrier")
            plt.gca().set_prop_cycle(None)
        plt.show()
     
    