from stable_baselines3 import PPO
from dynamic_reconfiguration_environment import NetEnv
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# ASCII VISUALIZATION (DEBUG / LOG FILE)
# ============================================================

def save_switch_timelines_ascii(env, filename="switch_timelines.txt", time_scale=10):
    lines = []
    lines.append("=== ASCII SWITCH TIMELINES ===\n")

    timelines = {}

    for link, ops in env.links_operations.items():
        if not ops:
            continue

        src, dst = link.link_id

        if env.graph.nodes[src]["node_type"] != "SW":
            continue

        port = f"{src}->{dst}"
        timelines.setdefault(src, {})
        timelines[src].setdefault(port, [])

        for flow, op in ops:
            timelines[src][port].append(
                (flow.flow_id, op.start_time, op.end_time)
            )

    for sw, ports in sorted(timelines.items()):
        lines.append(f"\n🔷 SWITCH {sw}")

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

                label = f"F{flow_id}"
                mid = s + (e - s) // 2
                for i, ch in enumerate(label):
                    idx = mid + i
                    if 0 <= idx < width:
                        line[idx] = ch

            lines.append(f"{port:<15} | {''.join(line)}")

        lines.append(" " * 17 + "time →")

    lines.append("\n=== DONE ===")

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✔ ASCII timelines saved to {filename}")


# ============================================================
# GUI VISUALIZATION (MAIN TOOL)
# ============================================================

def plot_switch_timelines_gui(json_file="final_schedule.json"):
    with open(json_file, "r") as f:
        timelines = json.load(f)

    fig, ax = plt.subplots(figsize=(16, 9))

    # ---- color per flow ----
    flow_colors = {}
    color_cycle = plt.cm.tab20.colors
    color_idx = 0

    y = 0
    yticks = []
    ylabels = []

    # ---- iterate switches ----
    for sw, data in timelines.items():

        # group by egress port
        port_groups = {}

        for entry in data["timeline"]:
            port = entry["egress_port"]
            port_groups.setdefault(port, []).append(entry)

        # ---- draw one line per port ----
        for port, entries in sorted(port_groups.items()):
            yticks.append(y)
            ylabels.append(f"{sw} | {port}")

            for e in entries:
                flow_id = e["flow_id"]
                start = e["start"]
                end = e["end"]

                if flow_id not in flow_colors:
                    flow_colors[flow_id] = color_cycle[color_idx % len(color_cycle)]
                    color_idx += 1

                ax.barh(
                    y,
                    end - start,
                    left=start,
                    height=0.6,
                    color=flow_colors[flow_id],
                    edgecolor="black"
                )

                # flow label inside bar
                ax.text(
                    start + (end - start) / 2,
                    y,
                    f"{flow_id}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )

            y += 1

        # space between switches
        y += 0.8

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Time")
    ax.set_title("TSN Switch Scheduling Timeline (Per-Port View)")

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # legend
    legend = [
        mpatches.Patch(color=c, label=f"{fid}")
        for fid, c in flow_colors.items()
    ]

    ax.legend(
        handles=legend,
        title="Flows",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("\n=== Creating environment ===")
    env = NetEnv(network_file="simple_line.json")
    obs, info = env.reset()

    print(f"Number of flows: {len(env.flows)}")

    print("\n=== Loading trained agent ===")
    model = PPO.load("Preprocess_model_resaved.zip", env=env)

    print("\n=== Running agent (DEBUG MODE) ===")

    step = 0
    done = False

    while not done:
        step += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"\n--- STEP {step} ---")
        print(f"Action: {action}")

        if step > 100:
            print("⚠️ Safety break")
            break

    # ========================================================
    # EXPORT + VISUALIZATION
    # ========================================================

    env.export_switch_timelines_to_json("final_schedule.json")
    save_switch_timelines_ascii(env, "switch_timelines.txt")

    print("\n=== Launching GUI ===")
    plot_switch_timelines_gui("final_schedule.json")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
