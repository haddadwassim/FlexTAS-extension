from env import NetEnv, Flow
from network_state import NetworkState

def main():
    # Load empty network state
    network_state = NetworkState.from_json("network_state.json")

    # Create environment
    env = NetEnv()

    # Define flows
    flow1 = Flow(
        flow_id="f1",
        path=[{"node": "sw1", "port": "px"}, {"node": "sw2", "port": "py"}],
        payload=100,
        period=1000,
        jitter=50,
        e2e_delay=1000
    )

    flow2 = Flow(
        flow_id="f2",
        path=[{"node": "sw1", "port": "py"}, {"node": "sw2", "port": "px"}],
        payload=200,
        period=800,
        jitter=30,
        e2e_delay=900
    )

    flow3 = Flow(
        flow_id="f3",
        path=[{"node": "sw1", "port": "px"}],
        payload=150,
        period=1200,
        jitter=20,
        e2e_delay=1100
    )

    # Schedule flows
    env.schedule_flows([flow1, flow2, flow3], network_state)

    # Print results
    print("\n=== Final Network Schedule ===")
    network_state.pretty_print()

if __name__ == "__main__":
    main()
