from stable_baselines3 import PPO
from dynamic_reconfiguration_environment import NetEnv

# Track flow status for printing
flow_status = {}

def print_flows(env, title):
    print(f"\n=== {title} ===")
    if not env.flows:
        print("No flows in the network.")
        return
    for f in env.flows:
        status = flow_status.get(f.flow_id, "original")
        print(f"Flow {f.flow_id}: {f.src_id} -> {f.dst_id}, path length {len(f.path)} [{status}]")

def main():
    global flow_status

    print("=== Creating environment ===")
    env = NetEnv(network_file="simple_line.json")
    obs, info = env.reset()

    # Initialize flow_status
    for f in env.flows:
        flow_status[f.flow_id] = "original"

    print_flows(env, "Original Flows")

    print("\n=== Loading trained agent ===")
    model = PPO.load("Preprocess_model_resaved.zip", env=env)

    # -----------------------------
    # Let the model take a step (optional)
    # -----------------------------
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print("\nStep 1: Agent took an action")

    # -----------------------------
    # Add flows
    # -----------------------------
    add_event = {"add": 2}
    env.reconfigure(add_event)
    # Update flow_status
    for f in env.flows[-2:]:
        flow_status[f.flow_id] = "added"

    print_flows(env, "After Adding 2 Flows")

    # -----------------------------
    # Remove flows
    # -----------------------------
    if len(env.flows) >= 2:
        remove_ids = [f.flow_id for f in env.flows[:2]]
        remove_event = {"remove": remove_ids}
        env.reconfigure(remove_event)
        for fid in remove_ids:
            flow_status[fid] = "removed"

        # Reset observation to prevent agent errors
        obs, info = env.reset()

        print_flows(env, "After Removing 2 Flows")

if __name__ == "__main__":
    main()
