from env import NetEnv
from network_state import NetworkState
from src.network.net import Network, generate_cev, generate_flows
from src.app.drl_scheduler import DrlScheduler


def main():
    # -----------------------
    # Build network
    # -----------------------
    graph = generate_cev()
    flows = generate_flows(graph, 2)
    network = Network(graph, flows)

    # -----------------------
    # Create scheduler (spawns subprocesses)
    # -----------------------
    scheduler = DrlScheduler(network)
    scheduler.load_model("best_model", "MaskablePPO")

    # -----------------------
    # Create env for inference
    # -----------------------
    env = NetEnv(network)
    #env.attach_agent(scheduler.model)

    # -----------------------
    # Network state
    # -----------------------
    network_state = NetworkState.from_json("network_state.json")

    # -----------------------
    # Schedule
    # -----------------------
    env.add_flows(flows, network_state)

    # Schedule all flows dynamically
    while env.flow_index < len(env.flows):
        done = False
        while not done:
            state = env._generate_state()
            action = scheduler.model.predict(state)[0]
            obs, reward, done, truncated, info = env.step(action)

    # Update network_state after scheduling
    env.update_network_state(network_state)
    network_state.to_json("network_state.json")

    network_state.pretty_print()




if __name__ == "__main__":
    main()
