# harvest_sed

## Setup


```
git clone https://github.com/Founding-Zero/SED_LLM
make install
```

## Usage

Use json to modify the config:

```
python harvest_sed/__init__.py --config '{"block_size": 10}'
```
Or use a config of your own in /config:
```
python harvest_sed/__init__.py --config $PWD/config/example_config.json

```
Or reuse an old config file by specifying the path:
```
harvest_sed --config $PWD/runs/<YYYY-MM-DD>---<HH-MM-SS>/config.json
```
Within your config, you can specify tracking preferences, experiment parameters, optimization settings, and more.
## General Experiment Settings
```
exp_name (str): 'apple_picking_game' #The name of this experiment.
seed (int): 1 #Seed of the experiment for reproducibility.
cuda (bool): True #If toggled, CUDA will be enabled by default.
voting_type (str): 'simple_mean' # Voting mechanism to use when deciding the objective of the Principal from the preferences of the agents.
selfishness_dist (str): 'selfish' # Distribution type for agent selfishness.
```
## Tracking Settings
```
track (bool): False # If toggled, this experiment will be tracked with Weights and Biases.
wandb_project_name (str): 'apple-picking-game' # The Weights and Biases project name.
wandb_entity (str): None # The entity (team) of the Weights and Biases project.
log_locally (bool): False # If True, logs will be saved locally.
log_file (str): None # File path to save logs locally.
capture_video (bool): True # Whether to capture videos of the agent performances.
video_freq (int): 20 # Capture video every how many episodes?
save_model (bool): True # Whether to save model parameters.
save_model_freq (int): 100 # Save model parameters every how many episodes?
```
## Optimization Settings
```
learning_rate (float): 2.5e-4 # The learning rate of the optimizer.
adam_eps (float): 1e-5 # Epsilon value for the Adam optimizer.
num_parallel_games (int): 1 # The number of parallel game environments.
num_frames (int): 4 # The number of game frames to stack together.
num_episodes (int): 100000 # The number of episodes to run.
episode_length (int): 1000 # The number of steps in an episode.
tax_annealment_proportion (float): 0.02 # Proportion of episodes over which to linearly anneal the tax cap multiplier.
sampling_horizon (int): 200 # The number of timesteps between policy update iterations.
tax_period (int): 50 # The number of timesteps tax periods last (at end of period tax values are updated and taxes applied).
anneal_tax (bool): True # Toggle tax cap annealing over an initial proportion of episodes.
anneal_lr (bool): True # Toggle learning rate annealing for policy and value networks.
```
## Reinforcement Learning Settings
```
algorithm (str): 'ppo' # The algorithm to use for the agents and Principal.
LLM (bool): False # Toggle for using a Large Language Model in place of a typical RL algorithm for the Principal.
gamma (float): 0.99 # The discount factor gamma.
gae_lambda (float): 0.95 # The lambda for the general advantage estimation.
minibatch_size (int): 128 # Size of minibatches when training policy network.
update_epochs (int): 4 # The number of epochs to update the policy.
norm_adv (bool): True # Toggle advantages normalization.
clip_coef (float): 0.2 # The surrogate clipping coefficient.
clip_vloss (bool): True # Toggle whether to use a clipped loss for the value function, as per the paper.
ent_coef (float): 0.01 # Coefficient of the entropy term.
vf_coef (float): 0.5 # Coefficient of the value function term.
max_grad_norm (float): 0.5 # The maximum norm for gradient clipping.
target_kl (float): None # The target KL divergence threshold.
```
