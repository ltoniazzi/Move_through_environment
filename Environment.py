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

    



    def plot_flow(self, peds=0, drag=30, size_target=20, size_obstacle=4, alpha=.3, groups=0, n_agents_left=0):
        n = peds[0].steps
        drag = peds[0].steps
        if groups == 0:
            plt.figure(figsize=(5,5), dpi=80)
            plt.title(f"{self.name}:Time: {int(np.floor(n/60))}.{n%60}min")
            plt.xlim(-20,100)
            plt.ylim(-20,100)
            for obstacle_ in self.obstacles_to_plot():
                plt.scatter(obstacle_[0], obstacle_[1], s=size_obstacle, label=f"Barrier")
                plt.gca().set_prop_cycle(None)
            for ped in peds:
                plt.scatter(ped.target[0],ped.target[1], s=size_target, label=f"{ped}", color="r")
            #plt.gca().set_prop_cycle(None)
            for ped in peds:
                plt.plot(ped.history[0,max(1,n-drag):n],ped.history[1,max(1,n-drag):n], alpha=alpha, label=f"{ped}", color="g")

        else:
            plt.figure(figsize=(5,5), dpi=80)
            plt.title(f"{self.name}:Time: {int(np.floor(n/60))}.{n%60}min")
            #plt.xlim(-20,100)
            #plt.ylim(-20,100)
            for obstacle_ in self.obstacles_to_plot():
                plt.scatter(obstacle_[0], obstacle_[1], s=size_obstacle, label=f"Barrier")
                plt.gca().set_prop_cycle(None)
            for i, ped in enumerate(peds):
                if i < n_agents_left:
                    color = "r"
                else:
                    color = "g"
                plt.scatter(ped.target[0],ped.target[1], s=size_target, label=f"{ped}", color=color)
            for i, ped in enumerate(peds):
                if i < n_agents_left:
                    color = "r"
                else:
                    color = "g"
                plt.plot(ped.history[0,max(1,n-drag):n],ped.history[1,max(1,n-drag):n], alpha=alpha, label=f"{ped}", color=color)




    def plot_scatter(self, peds=0, drag=30, size=3):
        n = peds[0].steps
        plt.figure(figsize=(5,5), dpi=80)
        plt.title(f"Time: {int(np.floor(n/60))}.{n%60}min")
        plt.xlim(-20,100)
        plt.ylim(-20,100)
        for obstacle_ in self.obstacles_to_plot():
            plt.scatter(obstacle_[0], obstacle_[1], s=4, label=f"Barrier")
            plt.gca().set_prop_cycle(None)
        for ped in peds:
            plt.scatter(ped.target[0],ped.target[1], s=20, label=f"{ped}")
        plt.gca().set_prop_cycle(None)
        for ped in peds:
            plt.scatter(ped.history[0,max(1,n-drag):n],ped.history[1,max(1,n-drag):n], s=size, label=f"{ped}")
        
    