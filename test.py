import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from dynamic_reconfiguration_environment import NetEnv

def main():
    print("\n=== Creating environment ===")
    env = DummyVecEnv([lambda: NetEnv()])

    # Just to inspect initial number of flows
    temp_env = NetEnv()
    print(f"Number of flows: {len(temp_env.flows)}")

    # ---- Load trained agent ----
    print("\n=== Loading trained agent ===")
    model = PPO.load("Preprocess_model_resaved.zip", env=env)

    obs = env.reset()
    done = False
    steps = 0

    # Access underlying NetEnv for debugging
    real_env = env.envs[0]

    print("\n=== Running agent (debug mode) ===")
    while not done:
        # Predict action from agent
        action, _states = model.predict(obs, deterministic=True)
        print(f"\nStep {steps+1}, predicted action: {action}")

        # Step environment
        obs, reward, terminated, info = env.step(action)
        done = terminated[0]
        steps += 1

        # Debug: print temp_operations
        if real_env.temp_operations:
            print(f"  Temp operations ({len(real_env.temp_operations)}):")
            for link, op in real_env.temp_operations:
                print(f"    Link: {link.link_id}, Operation: start={op.start_time}, end={op.end_time}, flow_id={getattr(op, 'flow_id', None)}")
        else:
            print("  Temp operations: empty")

    print(f"\nAgent ran for {steps} steps")

    # ---- Inspect switch schedules ----
    print("\n=== Inspecting switch schedules ===")
    schedules = real_env.get_switch_schedules()
    for sw, ops in schedules.items():
        print(f"\nSwitch {sw}: {len(ops)} ops")
        for op in ops[:5]:
            print(f"  Flow {op['flow_id']} [{op['start']} → {op['end']}] via {op['link']}")

    # ---- Inspect TEMP schedules ----
    print("\n=== Inspecting TEMP schedules ===")
    temp_schedules = real_env.get_temp_schedules()
    for sw, ops in temp_schedules.items():
        print(f"\nSwitch {sw}: {len(ops)} temp ops")
        for op in ops[:5]:
            print(f"  Flow {op['flow_id']} [{op['start']} → {op['end']}] via {op['link']}")

    total_ops = sum(len(ops) for ops in schedules.values())
    print(f"\nTotal scheduled operations across all switches: {total_ops}")
    if total_ops == 0:
        print("⚠️  WARNING: No scheduling detected. Agent may not have scheduled anything yet.")
    else:
        print("✅ Scheduling detected successfully.")

if __name__ == "__main__":
    main()
