from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import copy
import numpy as np
# from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import torch, hydra

# from scripts.reactive_tamp import REACTIVE_TAMP
# from src.m3p2i_aip.config.config_store import ExampleConfig
# import learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
# from sim1 import run_sim1

import random
import time


class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

    def clear(self):
        """Clear all trajectories and states"""
        self.pool.clear()
        self.pool_init_state.clear()
        self.counter = 0

class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.goal_distance = get_goal_distance(args)

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)

        self.sampler = None
        self.reactive_tamp = None

        # Subgoal learning parameters
        self.subgoal_dataset = []  # Store (state, goal, subgoal) tuples for training
        self.subgoal_dataset_capacity = 10000
        self.use_direct_subgoal = True
        self.subgoal_hindsight = True

        # Training status tracking
        self.training_state = {"total_episodes": 0}
        self.env = None
        self.env_test = None
        self.env_type = 'simpler'
        self.planner = None
        self.agent = None
        self.buffer = None

        # Return and performance tracking
        self.running_return_history = []
        self.running_return_avg = 0.0
        self.running_loss_history = []
        self.running_average_history = []
        self.progress_window_size = 30

        # Successful trajectory history
        self.success_history = []

        # All trajectory history (both success and failure)
        self.all_trajectories = []
        self.all_trajectories_capacity = 100
        self.all_episode_trajectories = []

        # Learning rate and early stopping
        self.best_return = -np.inf
        self.episodes_since_improvement = 0
        self.early_stop_patience = 100
        self.ema_return = None
        self.ema_alpha = 0.1
        self.save_best_model = True
        self.episodes = args.episodes
        self.cycles = 0

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Subgoal optimization settings
        self.use_subgoal_network = True

        # Progress window and EMA
        self.progress_window_size = 30
        self.ema_alpha = 0.1

        # Training stage tracking
        self.advanced_stage = False
        self.min_success_trajectories = 20

    def learn(self, args, env, env_test, agent, buffer, planner, training_state=None):
        """Main learning loop

        Args:
            args: configuration parameters
            env: training environment
            env_test: test environment
            agent: RL agent
            buffer: replay buffer
            planner: MPPI planner
            training_state: optional dictionary for tracking training state

        Returns:
            tuple: average return and return delta
        """
        self.initial_goals = []
        self.desired_goals = []
        self.explore_goals = []
        self.achieved_trajectories = []
        self.achieved_rewards = []
        self.episode_return = 0

        self.episode_trajectory = []
        # Set environment and components
        self.env = env
        self.env_test = env_test
        self.agent = agent
        self.buffer = buffer
        self.planner = planner

        # Update training state if provided
        if training_state is not None:
            self.training_state = training_state

        pre_return_avg = self.running_return_avg
        total_episodes = self.training_state.get("total_episodes", 0)

        current_epoch = self.training_state.get("current_epoch", 0)
        total_epochs = self.training_state.get("total_epochs", 1)

        exploration_epochs = int(total_epochs * 0.2)
        training_epochs = int(total_epochs * 0.5)

        if current_epoch < exploration_epochs:
            stage = "Exploration"
            stage_progress = current_epoch / exploration_epochs
        elif current_epoch < training_epochs:
            stage = "Training"
            stage_progress = (current_epoch - exploration_epochs) / (training_epochs - exploration_epochs)
        else:
            stage = "Fine-tuning"
            stage_progress = (current_epoch - training_epochs) / (total_epochs - training_epochs)

        check_interval = getattr(args, 'check_interval', 10)
        if total_episodes % check_interval == 0:
            print(f"[Epoch {current_epoch + 1}/{total_epochs}] Stage: {stage} (Progress: {stage_progress:.2f})")

        is_warmup = current_epoch < max(1, int(total_epochs * 0.25))
        self.is_warmup = is_warmup

        obs = self.env.reset()
        self.prev_position = obs['achieved_goal'].copy()
        goal_a = obs['achieved_goal'].copy()
        goal_d = obs['desired_goal'].copy()
        self.initial_goals.append(goal_a.copy())
        self.desired_goals.append(goal_d.copy())

        for episode in range(self.episodes):
            self.training_state["total_episodes"] = total_episodes + episode + 1

            achieved_goal = self.env._get_obs()['achieved_goal'].copy()
            desired_goal = self.env._get_obs()['desired_goal'].copy()

            if is_warmup:
                subgoal = achieved_goal + np.random.uniform(-0.08, 0.08, size=achieved_goal.shape)
                subgoal = np.clip(subgoal, -1.5, 1.5)
                print(f"Warmup phase: Using random subgoal {subgoal}")
            else:
                obs = self.env._get_obs()
                subgoal = self.generate_subgoal(obs, pretrain=is_warmup)

            timesteps = getattr(args, 'episode_duration', 50)
            episode_experience, episode_reward, trajectory, final_distance = self.rollout(timesteps, subgoal=subgoal)
            self.explore_goals.append(subgoal)
            self.episode_trajectory.append(trajectory)
            self.episode_return += episode_reward

            self.buffer.store_trajectory(episode_experience)
            self.record_return(episode_reward, episode)

            if episode % self.cycles == 0:
                self.update_network()
                if self.running_return_history:
                    avg_loss = np.mean(self.running_loss_history) if self.running_loss_history else 0
                    print(
                        f"Episode {self.training_state['total_episodes']}, Avg Return: {self.running_return_avg:.4f}, Avg Loss: {avg_loss:.4f}")

            if final_distance < 0.005:
                break

        final_trajectory = np.concatenate(self.episode_trajectory)

        self.achieved_trajectories.append(final_trajectory)
        self.all_episode_trajectories.append(final_trajectory)

        self.training_state["episodes"] = self.training_state.get("episodes", 0) + self.episodes
        return_delta = self.running_return_avg - pre_return_avg

        return self.running_return_avg, return_delta

    def train_subgoal_network(self):
        """Train subgoal generation network

        Two-stage training strategy:
        1. Early stage: sample from all historical trajectories, select points closer to the goal as subgoals
        2. Advanced stage: use value function to optimize subgoal selection based on successful trajectories
        """
        if not hasattr(self.agent, 'use_direct_subgoal') or not self.agent.use_direct_subgoal:
            return

        condition = len(self.all_episode_trajectories) > 400 or len(self.success_history) >= self.min_success_trajectories
        advanced_stage = condition and not self.is_warmup

        if advanced_stage and not self.advanced_stage:
            self.advanced_stage = True
            print(f"[Subgoal Training] Entered advanced training stage!")

        if advanced_stage:
            print(f"[Subgoal Training] Advanced stage: using {len(self.success_history)} successful trajectories for training")
        else:
            print(f"[Subgoal Training] Basic stage: sampling subgoals from {len(self.all_episode_trajectories)} historical trajectories")

        subgoal_data = {
            'obs': [],
            'goal': [],
            'subgoal_target': []
        }

        if not advanced_stage and len(self.all_episode_trajectories) > 0:
            valid_samples = 0
            desired_goal = self.env._get_obs()['desired_goal'].copy()

            trajectories = self.all_episode_trajectories
            if len(trajectories) > 150:
                distances = []
                for traj in trajectories:
                    if len(traj) < 8:
                        distances.append(np.inf)
                    else:
                        traj_end = traj[-1]
                        dist = np.linalg.norm(traj_end - desired_goal)
                        distances.append(dist)
                top_indices = np.argsort(distances)[:50]
                selected_trajectories = [trajectories[i] for i in top_indices]
            else:
                selected_trajectories = trajectories

            for traj_data in selected_trajectories:
                traj_obs = traj_data
                traj_length = len(traj_obs)
                if traj_length < 5:
                    continue

                for i in range(0, traj_length - 6, 2):
                    current_state = traj_obs[i].copy()
                    final_goal = self.env._get_obs()['desired_goal'].copy()
                    current_to_goal_dist = np.linalg.norm(current_state - final_goal)

                    for j in range(1, min(6, traj_length - i)):
                        future_idx = i + j
                        future_state = traj_obs[future_idx].copy()
                        future_to_goal_dist = np.linalg.norm(future_state - final_goal)

                        if 0.02 < current_to_goal_dist - future_to_goal_dist:
                            full_observation = np.zeros(13)
                            full_observation[:3] = current_state
                            complete_obs = np.concatenate([full_observation, final_goal])

                            subgoal_data['obs'].append(complete_obs)
                            subgoal_data['goal'].append(final_goal)
                            subgoal_data['subgoal_target'].append(future_state)
                            valid_samples += 1
                            break

            print(f"[Subgoal Training] Extracted {valid_samples} valid subgoal samples from historical trajectories")

        if advanced_stage:
            subgoal_data = {
                'obs': [], 'goal': [], 'subgoal_target': []
            }
            n_samples = min(200, len(self.success_history))
            if n_samples > 0:
                sampled_trajs = random.sample(self.success_history, n_samples)
                success_samples = 0

                for traj in sampled_trajs:
                    traj_length = len(traj['obs'])
                    if traj_length < 5:
                        continue

                    for i in range(0, traj_length - 10, 2):
                        current_state = traj['obs'][i]['achieved_goal'].copy()
                        final_goal = traj['obs'][i]['desired_goal'].copy()

                        current_to_goal_dist = np.linalg.norm(current_state - final_goal)

                        for j in range(1, min(10, traj_length - i)):
                            future_idx = i + j
                            future_state = traj['obs'][future_idx]['achieved_goal'].copy()
                            future_to_goal_dist = np.linalg.norm(future_state - final_goal)

                            if 0.03 < current_to_goal_dist - future_to_goal_dist:
                                subgoal_target = future_state
                                full_observation = np.zeros(13)
                                full_observation[:3] = current_state
                                complete_obs = np.concatenate([full_observation, final_goal])

                                subgoal_data['obs'].append(complete_obs)
                                subgoal_data['goal'].append(final_goal)
                                subgoal_data['subgoal_target'].append(subgoal_target)
                                success_samples += 1
                                break

                print(f"[Subgoal Training] Extracted {success_samples} subgoal samples from successful trajectories")

        if len(subgoal_data['obs']) < 10:
            print("[Subgoal Training] Not enough valid training data, skipping training")
            return

        for key in subgoal_data:
            subgoal_data[key] = np.array(subgoal_data[key])

        print(f"[Subgoal Training] Prepared {len(subgoal_data['obs'])} training samples")

        if len(subgoal_data['obs']) > 3000:
            recent_N = 1500
            batch_size = min(64, recent_N)
            start_idx = max(0, len(subgoal_data['obs']) - recent_N)
            recent_range = np.arange(start_idx, len(subgoal_data['obs']))
            n_batches = len(recent_range) // batch_size

            total_loss = 0
            for _ in range(n_batches):
                idxs = np.random.choice(recent_range, batch_size, replace=False)
                batch = {
                    'obs': np.array(subgoal_data['obs'])[idxs],
                    'goal': np.array(subgoal_data['goal'])[idxs],
                    'subgoal_target': np.array(subgoal_data['subgoal_target'])[idxs]
                }
                loss = self.agent.train_subgoal(batch)
                if loss is not None:
                    total_loss += loss
        else:
            batch_size = min(64, len(subgoal_data['obs']))
            n_batches = len(subgoal_data['obs']) // batch_size
            total_loss = 0
            for _ in range(n_batches):
                idxs = np.random.randint(0, len(subgoal_data['obs']), batch_size)
                batch = {
                    'obs': subgoal_data['obs'][idxs],
                    'goal': subgoal_data['goal'][idxs],
                    'subgoal_target': subgoal_data['subgoal_target'][idxs]
                }
                loss = self.agent.train_subgoal(batch)
                if loss is not None:
                    total_loss += loss

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Subgoal Training] Training complete, average loss: {avg_loss:.4f}")

        return {"subgoal_loss": avg_loss}


    def update_network(self):
        """Update both the policy and subgoal networks."""

        # Sample from the replay buffer
        transitions = self.buffer.sample_batch(self.args.batch_size)

        if transitions is None:
            print("Warning: Failed to sample from buffer, skipping update")
            return

        # Train the policy network
        info = self.agent.train(transitions)

        # If subgoal training is enabled, train the subgoal network
        if hasattr(self, 'use_subgoal_network') and self.use_subgoal_network:
            subgoal_info = self.train_subgoal_network()
            if subgoal_info is not None:
                for k, v in subgoal_info.items():
                    info[k] = v

        # Update the total episodes count
        if 'total_episodes' not in self.training_state:
            self.training_state['total_episodes'] = 0
        self.training_state['total_episodes'] += 1

        # Recalculate average return every check_interval episodes
        check_interval = getattr(self.args, 'check_interval', 10)
        if self.training_state.get("total_episodes", 0) % check_interval == 0:
            self.calculate_running_avg_return()

    def generate_subgoal(self, obs, pretrain=False):
        """Generate a subgoal (direct prediction instead of offset)

        Stage-based strategy:
        1. Early stage: generate simple subgoals without value optimization
        2. Advanced stage: optimize subgoals using value function

        Args:
            obs: current observation dictionary
            pretrain: whether the model is in the pretraining stage

        Returns:
            subgoal: generated subgoal
        """
        is_basic_stage = pretrain or not self.advanced_stage

        if is_basic_stage:
            subgoal = self.agent.step(obs, explore=not pretrain, goal_based=True)
            if isinstance(subgoal, torch.Tensor):
                subgoal = subgoal.cpu().numpy()
            return subgoal

        else:
            initial_subgoal = self.agent.step(obs, explore=False, goal_based=True)
            if isinstance(initial_subgoal, torch.Tensor):
                initial_subgoal = initial_subgoal.cpu().numpy()

            optimized_subgoal = self.optimize_subgoal_with_noise(
                obs['observation'], initial_subgoal, obs['desired_goal'],
                n_samples=15, noise_scale=0.005
            )
            return optimized_subgoal

    def rollout(self, timesteps, subgoal=None):
        """Execute a subgoal-guided trajectory

        Args:
            timesteps: maximum number of steps
            subgoal: optional subgoal

        Returns:
            episode_experience: trajectory experience
            episode_reward: accumulated reward
            trajectory: list of achieved positions
            final_distance: distance to desired goal at the end
        """
        self.env.goal = subgoal
        obs = self.env._get_obs()
        current = Trajectory(obs)
        trajectory = [obs['achieved_goal'].copy()]

        episode_reward = 0

        if subgoal is not None:
            self.env.subgoal = torch.tensor(subgoal, dtype=torch.float32)

        initial_position = obs['achieved_goal'].copy()
        desired_goal = obs['desired_goal'].copy()
        direct_distance_to_goal = np.linalg.norm(initial_position - desired_goal)

        total_path_length = 0.0

        for t in range(timesteps):
            achieved_goal = obs['achieved_goal'].copy()

            action_mppi = bytes_to_torch(
                self.planner.run_tamp(
                    torch_to_bytes(self.env._dof_state),
                    torch_to_bytes(self.env._root_state),
                    subgoal.tolist() if subgoal is not None else desired_goal.tolist())
            )

            obs, reward, done, info, distance, dis_subgoal = self.env.step(action_mppi)

            prev_distance = np.linalg.norm(self.prev_position - obs['desired_goal'].copy())
            current_pos = obs['achieved_goal'].copy()
            curr_distance = np.linalg.norm(current_pos - obs['desired_goal'].copy())
            distance_improvement = curr_distance - prev_distance
            reward_distance = distance_improvement * (-1)
            reward = reward_distance

            time_penalty = -0.02
            reward += time_penalty

            if distance < 0.05:
                success_bonus = 10
                reward += success_bonus

            step_distance = np.linalg.norm(current_pos - self.prev_position)
            total_path_length += step_distance
            self.prev_position = current_pos

            episode_reward += reward

            if subgoal is not None and isinstance(subgoal, np.ndarray):
                subgoal = torch.tensor(subgoal, dtype=torch.float32)
            current.store_step(action_mppi, obs, reward, done, subgoal)
            trajectory.append(current_pos)

            if dis_subgoal < 0.005:
                print("----------------------Reached subgoal-----------------", subgoal)
                break

        final_direct_distance = np.linalg.norm(trajectory[-1] - initial_position)
        final_efficiency = final_direct_distance / (total_path_length + 1e-6)

        trajectory_data = {
            'obs': current.ep['obs'],
            'path': np.array(trajectory),
            'efficiency': final_efficiency,
            'path_length': total_path_length,
            'reward': episode_reward,
            'success': False
        }

        final_distance = np.linalg.norm(trajectory[-1] - desired_goal)
        if final_distance < 0.1:
            trajectory_data['success'] = True
            self.success_history.append(trajectory_data)
            if len(self.success_history) > 200:
                self.success_history.pop(0)

            if len(self.success_history) >= self.min_success_trajectories and not self.advanced_stage:
                self.advanced_stage = True
                print(f"[Training Stage] Collected {len(self.success_history)} successful trajectories, entering advanced stage!")

        self.all_trajectories.append(trajectory_data)
        if len(self.all_trajectories) > self.all_trajectories_capacity:
            self.all_trajectories.pop(0)

        return current, episode_reward, trajectory, final_distance

    def record_return(self, episode_reward, episode_idx):
        """Record and update return statistics

        Args:
            episode_reward: cumulative reward of the current episode
            episode_idx: index of the episode
        """
        self.running_return_history.append(episode_reward)
        if len(self.running_return_history) > self.progress_window_size:
            self.running_return_history.pop(0)

        self.running_return_avg = np.mean(self.running_return_history)

        if self.ema_return is None:
            self.ema_return = episode_reward
        else:
            self.ema_return = self.ema_alpha * episode_reward + (1 - self.ema_alpha) * self.ema_return

        if self.running_return_avg > self.best_return:
            self.best_return = self.running_return_avg
            self.episodes_since_improvement = 0

            if episode_idx > 20 and hasattr(self, 'save_best_model') and self.save_best_model:
                try:
                    import os
                    os.makedirs("saved_models", exist_ok=True)
                    torch.save(self.agent.subgoal_network.state_dict(), "saved_models/subgoal_network.pth")
                    torch.save(self.agent.policy.state_dict(), "saved_models/best_policy.pth")
                    torch.save(self.agent.critic.state_dict(), "saved_models/best_critic.pth")
                    print(f"Saved new best model, avg return: {self.best_return:.4f}")
                except Exception as e:
                    print(f"Error saving model: {e}")
        else:
            self.episodes_since_improvement += 1

    def evaluate_subgoal_value(self, obs, subgoal, final_goal):
        """
        Evaluate the Q-value of a given subgoal

        Args:
            obs: [13] numpy array, current robot observation
            subgoal: [3] numpy array, candidate subgoal
            final_goal: [3] numpy array, final desired goal

        Returns:
            value: float, estimated Q-value
        """
        device = next(self.agent.critic.parameters()).device

        if isinstance(final_goal, np.ndarray):
            final_goal = torch.tensor(final_goal, dtype=torch.float32, device=device)
        if final_goal.ndim == 1:
            final_goal = final_goal.unsqueeze(0)

        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        subgoal = torch.tensor(subgoal, dtype=torch.float32, device=device).unsqueeze(0)

        if obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)

        state_final_goal = torch.cat([obs, final_goal], dim=1)
        state = torch.cat([state_final_goal, subgoal], dim=1)

        state = self.agent._state_preprocessor(state, train=True)

        action, _, _ = self.agent.policy.act({"states": state.T}, role="policy")
        q1, _, _ = self.agent.critic.act({"states": state.T, "taken_actions": action}, role="critic")
        q2, _, _ = self.agent.critic2.act({"states": state.T, "taken_actions": action}, role="critic")

        value = torch.min(q1, q2)

        return value.item()

    def optimize_subgoal_with_noise(self, current_state, predicted_subgoal, final_goal, n_samples=15, noise_scale=0.008):
        """Optimize a predicted subgoal using noise sampling

        Args:
            current_state: [13] numpy array, current observation
            predicted_subgoal: [3] numpy array, initial subgoal predicted by the model
            final_goal: [3] numpy array, final desired goal
            n_samples: int, number of noise samples to generate
            noise_scale: float, standard deviation of the Gaussian noise

        Returns:
            optimized_subgoal: subgoal with the highest evaluated value
        """
        candidates = [predicted_subgoal]
        values = [self.evaluate_subgoal_value(current_state, predicted_subgoal, final_goal)]

        for _ in range(n_samples):
            noise = np.random.normal(0, noise_scale, size=predicted_subgoal.shape)
            noisy_subgoal = np.clip(predicted_subgoal + noise, -1.5, 1.5)
            value = self.evaluate_subgoal_value(current_state, noisy_subgoal, final_goal)

            candidates.append(noisy_subgoal)
            values.append(value)

        best_idx = np.argmax(values)
        best_subgoal = candidates[best_idx]
        best_value = values[best_idx]

        if best_idx > 0:
            improvement = best_value - values[0]
            print(f"Noise optimization improvement: {improvement:.4f}, before: {values[0]:.4f}, after: {best_value:.4f}")

        return best_subgoal

