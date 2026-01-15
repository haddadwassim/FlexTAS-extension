from env import NetEnv, Flow
from network_state import NetworkState

def main():
    # Initialize env and network state
    env = NetEnv()
    network_state = NetworkState.from_json("network_state.json")

    # Example flows to schedule
    flow1 = Flow(
        flow_id="f2",
        path=[{"node": "sw1", "port": "px"}],
        payload=500,
        period=1000,
        jitter=100,
        e2e_delay=1000,
        qos={"max_delay_us": 1000}
    )

    flow2 = Flow(
        flow_id="f3",
        path=[{"node": "sw1", "port": "px"}],
        payload=300,
        period=800,
        jitter=50,
        e2e_delay=900,
        qos={"max_delay_us": 800}
    )

    # Schedule flows
    env.schedule_flows([flow1, flow2], network_state)

    # Print updated network state
    network_state.pretty_print()

if __name__ == "__main__":
    main()
