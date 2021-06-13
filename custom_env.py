import os
import numpy as np
import gym
import tempfile
import xml.etree.ElementTree as ET
from sb3.base import GoalEnv

class Reach(GoalEnv):
    TARGET_SIZE = None
    ARENA_SPACE_LOW = None
    ARENA_SPACE_HIGH = None
    TARGET_DIM = None # The current location in ref to target is skill_obs[:TARGET_DIM] = obs[AGENT_DIM:AGENT_DIM+SKILL_DIM][:TARGET_DIM]
    REWARD_SCALE = None
    SPARSE_REWARD = None
    SURVIVE_REWARD = 0
    VISUALIZE = False

    def __init__(self, model_path=None):
        self.target = self.ARENA_SPACE_HIGH # Init to far away from agent start.
        self.num_reached = 0
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        if self.VISUALIZE:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
            tree = ET.parse(xml_path)
            world_body = tree.find(".//worldbody")
            _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
            target_elem = ET.Element('body')
            target_elem.set("name", "target")
            target_elem.set("pos", "0 0 " + str(self.TARGET_SIZE))
            target_geom = ET.SubElement(target_elem, "geom")
            target_geom.set("conaffinity", "0")
            target_geom.set("contype", "0")
            target_geom.set("name", "target")
            target_geom.set("pos", "0 0 0")
            target_geom.set("rgba", "0.2 0.9 0.2 0.8")
            target_geom.set("size", str(self.TARGET_SIZE))
            target_geom.set("type", "sphere")
            world_body.insert(-1, target_elem)
            tree.write(xml_path)
        else:
            xml_path = None
        
        super(Reach, self).__init__(model_path=xml_path)

    def _get_obs(self):
        return NotImplemented

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist_to_target = np.linalg.norm(achieved_goal - desired_goal)
        reward = -1*self.REWARD_SCALE * dist_to_target
        if dist_to_target < self.TARGET_SIZE:
            reward += self.SPARSE_REWARD
        reward += self.SURVIVE_REWARD
        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        desired_goal = self.target
        achieved_goal = self.skill_obs(obs)[:self.TARGET_DIM]
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = False
        if np.linalg.norm(achieved_goal - desired_goal) < self.TARGET_SIZE:
            done = True
        return obs, reward, done, {'success' : done}
        
class ReachNav(Reach):
    ARENA_SPACE_LOW = np.array([-8.0, -8.0])
    ARENA_SPACE_HIGH = np.array([8.0, 8.0])
    SKILL_DIM = 2
    TARGET_DIM = 2
    TASK_DIM = 4 # agent position, target position
    TARGET_SIZE = 0.5
    REWARD_SCALE = 0.1
    SPARSE_REWARD = 50
    VISUALIZE = True

class Reach_PointMass(ReachNav):
    ASSET = 'point_mass.xml'
    AGENT_DIM = 2
    FRAME_SKIP = 3

    def _get_obs(self):
        return {
            'observation' : np.concatenate((self.sim.data.qvel.flat[:],
                                            self.get_body_com("torso")[:2]), axis=0),
            'achieved_goal' : self.get_body_com("torso")[:2],
            'desired_goal' : self.target,
        }

    def reset(self):
        self.target = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.TARGET_DIM] = self.target
        qpos = self.init_qpos + self.np_random.uniform(low=-self.TARGET_SIZE, high=self.TARGET_SIZE, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()