import os
import numpy as np
import pandas as pd
import networkx as nx

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from Preprocess_environment import NetEnv
from src.network.net import generate_flows, Network

# ================================================
# CONFIGURATION
# ================================================

# Output paths
log_dir = "out/ppo_sorting_training"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.csv")
reward_file = os.path.join(log_dir, "reward_log.csv")
final_model_file = os.path.join(log_dir, "ppo_sorting_model.zip")

# Graph to use
graph_dir = "graphs"
graph_file = os.path.join(graph_dir, "ERG.graphml")
fixed_graph = nx.read_graphml(graph_file)

# Curriculum flow levels
flow_counts = [10, 30, 50, 100, 150, 200, 250, 300]
steps_per_level = 100_000

# PPO parameters
ppo_batch_size = 2048
learning_rate = 1e-4

# ================================================
# TRAINING LOOP
# ================================================

timesteps_done = 0
log_records = []
reward_records = []
model = None

for num_flows in flow_counts:
    print(f"\n=== Starting training with {num_flows} flows ===")

    # Generate flows for this flow level
    flows = generate_flows(
        fixed_graph,
        num_flows=num_flows,
        period_set=[2000, 4000, 8000, 16000, 32000, 64000, 128000],
        jitters=[0.1, 0.2, 0.5]
    )

    # Create environment
    network = Network(fixed_graph, flows)
    net_env = NetEnv(network)
    net_env.fixed_flows = True

    env = DummyVecEnv([lambda: Monitor(net_env)])

    if model is None:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=ppo_batch_size,
            verbose=2,
        )
    else:
        model.set_env(env)

    steps_this_level = 0

    while steps_this_level < steps_per_level:
        remaining = steps_per_level - steps_this_level
        steps = min(ppo_batch_size, remaining)

        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        steps_this_level += steps
        timesteps_done += steps

        scheduled_flows = set()
        for link_ops in net_env.links_operations.values():
            for flow, _ in link_ops:
                scheduled_flows.add(flow.flow_id)

        num_scheduled = len(scheduled_flows)
        print(f"Step {timesteps_done}: Flows={num_flows}, Scheduled={num_scheduled}/{num_flows}")

        log_records.append({
            "timesteps_done": timesteps_done,
            "flows": num_flows,
            "scheduled": num_scheduled,
            "using_rbc": False,
        })

        if hasattr(net_env, 'episode_rewards') and net_env.episode_rewards:
            recent_reward = np.mean(net_env.episode_rewards[-20:])
        else:
            recent_reward = 0.0

        reward_records.append({
            "timesteps_done": timesteps_done,
            "flows": num_flows,
            "avg_recent_reward": recent_reward
        })

    model_path = os.path.join(log_dir, f"ppo_model_{num_flows}_flows.zip")
    model.save(model_path)
    print(f"Saved model after {num_flows} flows to {model_path}")

model.save(final_model_file)
print(f" Saved FINAL PPO model to {final_model_file}")

pd.DataFrame(log_records).to_csv(log_file, index=False)
pd.DataFrame(reward_records).to_csv(reward_file, index=False)
print(f" Training log saved to {log_file}")
print(f" Reward log saved to {reward_file}")
