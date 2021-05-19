import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
from gym.utils import EzPickle, seeding
from gym.wrappers import TimeLimit
import mujoco_py
from mujoco_py import MjRenderContextOffscreen
import gym
import os
from collections import OrderedDict, deque
from gym import spaces 
DEFAULT_SIZE = 84
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchImageEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, fixed):


        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.fixed = fixed
        

        super().__init__(
                model_path=model_path, n_substeps=n_substeps, n_actions=4,
                initial_qpos=initial_qpos)
        obs= self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            image_observation=spaces.Box(0, 255, shape=obs['image_observation'].shape, dtype='uint8'),
            desired_goal_image=spaces.Box(0, 255, shape=obs['desired_goal_image'].shape, dtype='uint8')
        ))

    # required function 
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    
    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        # state_goal = self.goal.copy() 
        

        goal_image = self.get_goal_image(grip_pos=grip_pos.copy())
        current_image = self.get_image()
        assert (self.sim.data.get_site_xpos('robot0:grip') == grip_pos).all(), "gripper positon has not changed after taking goal image"
    
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal,
            "image_observation": current_image, # achieved goal can be derived from observation
            "desired_goal_image":goal_image
        }

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            if not self.fixed:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            else:
                goal = self.goal       
        return goal.copy()
        # desired_goal= goal.copy()
        # # move gripper to goal , take image and reset()
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip') # current gripper state
        # self.sim.data.set_mocap_pos('robot0:mocap', desired_goal)
        # for _ in range(10):
        #     self.sim.step()
        # self.goal_image =  self.get_image()

        # self.sim.data.set_mocap_pos('robot0:mocap', grip_pos)
        # for _ in range(10):
        #     self.sim.step()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_goal_image(self, grip_pos):
        desired_goal= self.goal
        # move gripper to goal , take image and reset()
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip') # current gripper state
        self.sim.data.set_mocap_pos('robot0:mocap', desired_goal)
        for _ in range(10):
            self.sim.step()
        
        # self._render_callback() = None
        
        
        goal_image =  self.get_image() 

        self.sim.data.set_mocap_pos('robot0:mocap', grip_pos)
        for _ in range(10):
            self.sim.step()

        return goal_image.copy()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
        # get goal image
        # it has to in _sample_goal as goals change at every reset 
        # think about render_Callback 
        self.goal= self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)

    def _render_callback(self):

            # Visualize target.
        
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

            
    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2]
                # if not self.fixed:
                object_xpos+=self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True
        
    # def load_viewer(self, device_id=-1):
    #     viewer = MjRenderContextOffscreen(self.sim, device_id=device_id)
    #     viewer.cam.distance = 1.4 
    #     viewer.cam.azimuth = 180 
    #     viewer.cam.elevation = -25
    #     # viewer.cam.lookat[2] = 0.5 
    #     viewer.cam.lookat[2] = 0.55
    #     viewer.cam.lookat[1] = 0.75
        return viewer
    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        # for idx, value in enumerate(lookat):
        #     self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -25.
        self.viewer.cam.lookat[2] = 0.55
        self.viewer.cam.lookat[1] = 0.75
        # self.viewer = self.load_viewer()



    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def render(self, mode='rgb_array', width=84, height=84):
        return super().render(mode, width, height)

    def get_image(self, width=84, height=84, camera_name=None): #use env.render("rgb_array") for marker to show up
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.sim.data.site_xpos[0]
        self.sim.forward()
        # viewer = self._get_viewer("rgb_array")
        
        # self._viewer_setup()
        # data= self.sim.render(
        #     width=width,
        #     height=height,
        #     camera_name=camera_name,
        # )
        self._get_viewer("rgb_array").render(width,height)
        data = self._get_viewer("rgb_array").read_pixels(width, height, depth=False)
        assert data is not None,"sim.render is None"
        return data[::-1,:,:]
    



class FrameStack(gym.Wrapper):
    """Stack frames for observation , goal_image ? """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.goal_frames=deque([], maxlen=k)
        wrapped_observation_space= env.observation_space
        self.observation_space=OrderedDict()
        pixels_spec = wrapped_observation_space['image_observation']
        self.observation_space['image_observation'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=np.concatenate([[pixels_spec.shape[2] * k], pixels_spec.shape[:2]], axis=0),
            dtype=env.observation_space['image_observation'].dtype
        )
        self.observation_space['desired_goal_image'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=np.concatenate([[pixels_spec.shape[2] * k], pixels_spec.shape[:2]], axis=0),
            dtype=env.observation_space['desired_goal_image'].dtype
        )
        
        
        self.observation_space['observation']= wrapped_observation_space['observation']
        self.observation_space['desired_goal'] = wrapped_observation_space['desired_goal']
        self.observation_space['achieved_goal']= wrapped_observation_space['achieved_goal']

    def reset(self): 
          ##TODO  framestack goal image? 
        obs = self.env.reset()
        image_observation = obs['image_observation'].transpose(2, 0, 1).copy()
        desired_goal_image = obs['desired_goal_image'].transpose(2, 0, 1).copy()
        for _ in range(self._k):
            self._frames.append(image_observation)
            self.goal_frames.append(desired_goal_image)
        return self._transform_observation(obs)

    def step(self, action): 
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs['image_observation'].transpose(2,0,1).copy())
        self.goal_frames.append(obs['desired_goal_image'].transpose(2,0,1).copy())
        # print(obs)
        return self._transform_observation(obs), reward, done, info

    def _transform_observation(self, observation):
        assert len(self._frames) == self._k
        assert len(self.goal_frames) == self._k
        obs = OrderedDict()
        obs['observation'] = observation['observation']
        obs['achieved_goal'] = observation['achieved_goal']
        obs['desired_goal'] = observation['desired_goal']
        obs['image_observation'] = np.concatenate(list(self._frames), axis=0)
        # obs['desired_goal_image'] = observation['desired_goal_image'].transpose(2,0,1).copy()
        
        obs['desired_goal_image'] = np.concatenate(list(self.goal_frames), axis=0) 
        # obs= spaces.Dict(obs)
        observation.update(obs)
        return observation

class ActionRepeat(gym.Wrapper):
    # def __init__(self, env, amount):
    #     gym.Wrapper.__init__(self, env)
    #     self._amount= amount
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount= amount
    
    def step(self, action):
        reward = 0.0
        first_time_step =None
        for i in range(self._amount):
            time_step = list(self.env.step(action))
            if i == 0 :
                first_time_step= time_step
            reward = reward+ time_step[1]
            if time_step[2]:
                break
        
        first_time_step[1] = reward
        return tuple(first_time_step)



MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')
REWARD_TYPE = "sparse"
class FetchReachImageEnv(FetchImageEnv,EzPickle):   
    def __init__(self, reward_type=REWARD_TYPE, fixed=True):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchImageEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,fixed=fixed)
        EzPickle.__init__(self)




class FetchSlideImageEnv(FetchImageEnv,EzPickle):
    def __init__(self, reward_type=REWARD_TYPE,fixed=True):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        FetchImageEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, fixed=fixed)


        EzPickle.__init__(self)


        
def make(env_name, frame_stack, action_repeat,max_episode_steps=50, seed=5,fixed=True, reward_type="sparse"):


    if env_name == 'fetch_reach':
        reach_env = FetchReachImageEnv(reward_type=reward_type,fixed=fixed)
        reach_env = ActionRepeat(reach_env,amount=action_repeat)
        reach_env = FrameStack(reach_env,k=frame_stack)
        reach_env = TimeLimit(reach_env,max_episode_steps=max_episode_steps)
    
    # else:
    #     env = suite.load(domain,
    #                      task,
    #                      task_kwargs={'random': seed},
    #                      visualize_reward=False)

    # apply action repeat and scaling
    # reach_env = ActionRepeatGymWrapper(reach_env, action_repeat)
    
    # reach_env = FrameStack(reach_env, frame_stack)
    print(reach_env.seed(seed))
    
    return reach_env


# from gym.envs.registration import registry, register, make, spec
# register(
#         id='FetchReachImageEnv-{}-v0'.format(REWARD_TYPE),
#         entry_point='fetch_image_env:FetchReachImageEnv',
#         max_episode_steps=50,
#     )

## TODO Check _reset_sim for has_object later for fixed goal setting / rendering 
## TODO find out if render or get_image is faster
