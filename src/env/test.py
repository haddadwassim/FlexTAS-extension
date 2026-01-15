from network_state import NetworkState


def main():
    # Load network state from JSON
    state = NetworkState.from_json("network_state.json")

    # Pretty print current schedule
    state.pretty_print()


if __name__ == "__main__":
    main()

