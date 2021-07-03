import os
import numpy as np
from base import GoalEnv
from gym.wrappers import TimeLimit
import gym
from gym.envs.robotics import utils
class Reach(GoalEnv):
    TARGET_SIZE = None
    ARENA_SPACE_LOW = None
    ARENA_SPACE_HIGH = None
    TARGET_DIM = None # The current location in ref to target is skill_obs[:TARGET_DIM] = obs[AGENT_DIM:AGENT_DIM+SKILL_DIM][:TARGET_DIM]
    REWARD_SCALE = None
    SPARSE_REWARD = None
    SURVIVE_REWARD = 0
    VISUALIZE = False

    def __init__(self,initial_qpos,  seed=None,model_path=None):
        self.target = self.ARENA_SPACE_HIGH # Init to far away from agent start.
        self.num_reached = 0
        if model_path is None:
            model_path = self.ASSET
#         xml_path = os.path.join("./assets", self.ASSET)
        
        self.init_site=None
        super(Reach, self).__init__(initial_qpos=initial_qpos, model_path=model_path)

    def get_obs(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist_to_target = np.linalg.norm(achieved_goal - desired_goal)
        reward = -1*self.REWARD_SCALE * dist_to_target
        if dist_to_target < self.TARGET_SIZE:
            reward += self.SPARSE_REWARD
        reward += self.SURVIVE_REWARD
        return reward
    

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        desired_goal = self.target
        achieved_goal = obs['achieved_goal']
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = False
        if np.linalg.norm(achieved_goal - desired_goal) < self.TARGET_SIZE:
            done = True
        return obs, reward, done, {'success' : done}
        
class ReachNav(Reach):
    def __init__(self, initial_qpos,seed):
        self.SKILL_DIM = 2
        self.TARGET_DIM = 2
        self.TASK_DIM = 4 # agent position, target position
        self.TARGET_SIZE = 0.6
        self.REWARD_SCALE = 0.1
        self.SPARSE_REWARD = 50
        self.VISUALIZE = True
        self.target_range=5
        
        super(ReachNav, self).__init__(initial_qpos,seed)

class Reach_PointMass(ReachNav):
    
    
    def __init__(self, initial_qpos,seed=10 ,fixed=False):
        self.ASSET = 'point_mass.xml'
        self.AGENT_DOF = 2
        self.FRAME_SKIP = 3
        self.fixed=fixed
        
        super(Reach_PointMass, self).__init__(initial_qpos,seed)
        self.init_site = self.sim.model.site_pos[0].copy()
        
        
   
    def viewer_setup(self):
        self.viewer.cam.distance = 14
        self.viewer.cam.azimuth = 90.
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[0] = -0.05821135
        self.viewer.cam.lookat[1] = 0.
        self.viewer.cam.lookat[2] = 0.5
        #pass
    def get_image(self, width=400, height=400):
        # move the target outside the camera frame 
        site_id = self.sim.model.site_name2id('target0')
        if self.init_site is not None :
            
            self.sim.model.site_pos[site_id] =  self.init_site 
            self.sim.forward()
        self._get_viewer("rgb_array").render(width,height)
        data = self._get_viewer("rgb_array").read_pixels(width, height, depth=False)
        assert data is not None,"sim.render is None"
        return data[::-1,:,:]
    

    
    def get_goal_image(self):
        desired_goal= self.target
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip') # current gripper state
        # move point_mass to target , take image and move back
        
        qpos = self.sim.get_state().qpos
        qvel = self.sim.get_state().qvel
       
        self.set_state(self.target, qvel)
        self.sim.forward()
#         for _ in range(10):
#             self.sim.step()
        
        # self._render_callback() = None
        
        
        goal_image =  self.get_image() 
        
        self.set_state(qpos, qvel)
        self.sim.forward()
#         for _ in range(10):
#             self.sim.step()
        return goal_image.copy()
    
    def get_obs(self):
        return {
            'observation' : np.concatenate((self.sim.data.qvel.flat[:],
                                            self.get_body_com("torso")[:2]), axis=0),
            
            'achieved_goal' : self.get_body_com("torso")[:2],
            'desired_goal' : self.target,
            'image_observation': self.get_image(),
            'desire_goal_image': self.get_goal_image()
            
            
        }
        
    
    def render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id][:2]= self.target - sites_offset[0][:2]
        self.sim.forward()
        pass


    
        ## bring the site to the target location 
    def reset(self):
        # randomize target
       
        
        qpos = self.init_qpos + self.np_random.uniform(low=-self.TARGET_SIZE, high=self.TARGET_SIZE, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.4, size=self.model.nv)
        self.set_state(qpos, qvel)
        
        
        if not self.fixed:
            
            self.target=  self.np_random.uniform(low=-self.target_range, high=self.target_range, size=2)
            self.target += self.sim.data.qpos.ravel().copy()
        return self.get_obs()

    
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
#         utils.reset_mocap_welds(self.sim)
        # calling forward rewrite the values in self.data 
        self.sim.forward()
#         self.initial_pos = self.init_qpos + self.np_random.uniform(low=-0.2, high=0.2, size=self.AGENT_DOF)
        self.target= self.np_random.uniform(low=-self.target_range, high=self.target_range, size=2)




def make_sb3_point_env(seed=0):

    initial_qpos = {
            'ballx': 0,
            'bally': 0
            
        }
    point_env = Reach_PointMass(initial_qpos,seed=seed)
    point_env = TimeLimit(point_env,max_episode_steps=100)
    return point_env