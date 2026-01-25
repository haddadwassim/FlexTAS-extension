from stable_baselines3 import PPO
from dynamic_reconfiguration_environment import NetEnv
import math


# ============================================================
# ASCII VISUALIZATION (DEBUG / ANALYSIS ONLY)
# ============================================================

def save_switch_timelines_ascii(env, filename="switch_timelines.txt", time_scale=10):
    """
    Save pretty ASCII timeline per switch / egress port into a text file.
    time_scale = how many time units per character
    """
    lines = []
    lines.append("=== ASCII SWITCH TIMELINES ===\n")

    # Build: switch -> port -> list of (flow_id, start, end)
    timelines = {}

    for link, ops in env.links_operations.items():
        if not ops:
            continue

        src, dst = link.link_id

        # only switches own schedules
        if env.graph.nodes[src]["node_type"] != "SW":
            continue

        port = f"{src}->{dst}"
        timelines.setdefault(src, {})
        timelines[src].setdefault(port, [])

        for flow, op in ops:
            timelines[src][port].append(
                (flow.flow_id, op.start_time, op.end_time)
            )

    # Render per switch
    for sw, ports in sorted(timelines.items()):
        lines.append(f"\n🔷 SWITCH {sw}")

        # compute global max time for alignment
        max_t = 0
        for ops in ports.values():
            for _, _, end in ops:
                max_t = max(max_t, end)

        width = math.ceil(max_t / time_scale) + 5

        for port, ops in sorted(ports.items()):
            line = [" "] * width

            for flow_id, start, end in ops:
                s = int(start / time_scale)
                e = max(s + 1, int(end / time_scale))

                for i in range(s, min(e, width)):
                    line[i] = "█"

                # label in the middle if possible
                label = f"{flow_id}"
                mid = s + (e - s) // 2
                for i, ch in enumerate(label):
                    idx = mid + i
                    if 0 <= idx < width:
                        line[idx] = ch

            lines.append(f"{port:<15} | {''.join(line)}")

        lines.append(" " * 17 + "time →")

    lines.append("\n=== DONE ===")

    # write to file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✔ ASCII timelines saved to {filename}")


# ============================================================
# MAIN TEST
# ============================================================

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

        if step > 100:
            print("⚠️ Safety break")
            break

    # ========================================================
    # EXPORT + VISUALIZATION TO FILE
    # ========================================================

    env.export_switch_timelines_to_json("final_schedule.json")
    save_switch_timelines_ascii(env, filename="switch_timelines.txt")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
