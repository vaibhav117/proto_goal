from tqdm.auto import tqdm
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import wandb

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()





class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1, gradient_save_freq=10, wandb_watch=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.gradient_save_freq = gradient_save_freq
        self.wandb_watch = wandb_watch
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.wandb_watch:
            
          d = {}
          for key in self.model.__dict__:
              if key in wandb.config:
                  continue
              if type(self.model.__dict__[key]) in [float, int, str]:
                  d[key] = self.model.__dict__[key]
              else:
                  d[key] = str(self.model.__dict__[key])
          if self.gradient_save_freq > 0:
              wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
          wandb.config.update(d)

            

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              
              wandb.log({"training_mean_reward": mean_reward.item()})
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True



class EvalCallback(BaseCallback):
  """
  Callback for evaluating an agent.
  
  :param eval_env: (gym.Env) The environment used for initialization
  :param n_eval_episodes: (int) The number of episodes to test the agent
  :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
  """
  def __init__(self, eval_env, n_eval_episodes=5, eval_freq=1000, log_dir=None):
    super(EvalCallback, self).__init__()
    self.eval_env = eval_env
    self.n_eval_episodes = n_eval_episodes
    self.eval_freq = eval_freq
    self.log_dir = log_dir
    self.save_best = os.path.join(log_dir, 'best_model')
    self.save_latest = os.path.join(log_dir, "latest_model")
    self.best_mean_reward = -np.inf

  def _init_callback(self) -> None:

        # Create folder if needed
    if self.log_dir is not None:
      os.makedirs(self.log_dir, exist_ok=True)
  
  def _on_step(self):
    """
    This method will be called by the model.

    :return: (bool)
    """
    # self.logger.record("current_reward")
    # self.n_calls is automatically updated because
    # we derive from BaseCallback
    if self.n_calls % self.eval_freq == 0:
      # === YOUR CODE HERE ===#
      # Evaluate the agent:
      # you need to do self.n_eval_episodes loop using self.eval_env
      # hint: you can use self.model.predict(obs, deterministic=True)
      mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
      # Save the latest agent
      self.logger.record("eval_mean_reward", mean_reward)
      self.model.save(self.save_latest)
      # and update self.best_mean_reward
      if mean_reward > self.best_mean_reward:
        self.best_mean_reward = mean_reward
        self.model.save(self.log_dir)
        if self.verbose > 0:
          print("Saving new best model at {} timesteps".format(self.n_calls))
          print("Saving new best model to {}.zip".format(self.save_best))
          
        print("Best mean reward: {:.2f}".format(self.best_mean_reward))
      

      # ====================== #    
    return True




# cant use wandbcallback for training 

class WandbCallback(BaseCallback):

  """ Log SB3 experiments to Weights and Biases
      - Added model tracking and uploading

    - Added complete hyperparameters recording
      - Added gradient logging
      - Note that `wandb.init(...)` must be called before the WandbCallback can be used
  Args:
      verbose: The verbosity of sb3 output
      model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
      model_save_freq: Frequency to save the model
      gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
  """

  def __init__(
        self,
        verbose: int = 0,
        gradient_save_freq: int = 0,
    ):

    super(WandbCallback, self).__init__(verbose)
    assert (wandb.run is not None), "no wandb run detected; use `wandb.init()` to initialize a run"
    self.gradient_save_freq = gradient_save_freq
        # Create folder if needed

  def _init_callback(self) -> None:


    d = {}
    for key in self.model.__dict__:
        if key in wandb.config:
            continue
        if type(self.model.__dict__[key]) in [float, int, str]:
            d[key] = self.model.__dict__[key]
        else:
            d[key] = str(self.model.__dict__[key])
    if self.gradient_save_freq > 0:
        wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
    wandb.config.update(d)

