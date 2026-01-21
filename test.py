from stable_baselines3 import PPO
from dynamic_reconfiguration_environment import NetEnv

def main():
    print("\n=== Creating environment ===")
    env = NetEnv()
    obs, info = env.reset()

    print(f"Number of flows: {len(env.flows)}")

    print("\n=== Loading trained agent ===")
    model = PPO.load("Preprocess_model_resaved.zip", env=env)

    print("\n=== Running agent (SIMPLE DEBUG) ===")
    step = 0
    done = False

    while not done:
        step += 1

        # Get the action from the agent
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"\n--- STEP {step} ---")
        print(f"Action: {action}")

        # 1️⃣ TEMP OPERATIONS
        if env.temp_operations:
            print(f"Temp ops ({len(env.temp_operations)}):")
            for link, op in env.temp_operations:
                u, v = link.link_id
                print(
                    f"  TEMP {u}->{v} "
                    f"[{op.start_time} → {op.end_time}]"
                )
        else:
            print("Temp ops: NONE")

        # 2️⃣ COMMITTED OPERATIONS (RAW, PER LINK)
        total_committed = sum(len(ops) for ops in env.links_operations.values())
        print(f"Total committed ops so far: {total_committed}")

        for link, ops in env.links_operations.items():
            if not ops:
                continue

            u, v = link.link_id
            owner_type = env.graph.nodes[u]["node_type"]

            print(f"  LINK {u}->{v} (owner: {owner_type})")
            for flow, op in ops:
                print(
                    f"    Flow {flow.flow_id} "
                    f"[{op.start_time} → {op.end_time}]"
                )

        # Safety break to avoid infinite loops
        if step > 100:
            print("⚠️ Safety break")
            break

    # ✅ Export switch timelines to JSON (using the method now in NetEnv)
    env.export_switch_timelines_to_json("final_schedule.json")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
