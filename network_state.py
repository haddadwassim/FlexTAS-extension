import json
from typing import Dict, List, Any


class NetworkState:
    """
    Stores the CURRENT scheduled state of the network.
    This includes:
      - Network topology
      - Already scheduled flows
      - Per-node / per-port schedules

    No scheduling decisions are made here.
    """

    def __init__(self, state: Dict[str, Any]):
        self.network = state.get("network", {})
        self.nodes = self.network.get("nodes", {})
        self.links = self.network.get("links", [])

        self.flows = state.get("flows", {})
        self.schedule = state.get("schedule", {})
        self.time = state.get("time", {})

        self._build_ports_from_links()
        self._validate_basic_structure()
        self.validate_references()

    # ----------------------------
    # Loading / Saving
    # ----------------------------

    @classmethod
    def from_json(cls, path: str) -> "NetworkState":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network,
            "flows": self.flows,
            "schedule": self.schedule,
            "time": self.time
        }

    # ----------------------------
    # Read-only helpers
    # ----------------------------

    def get_nodes(self) -> List[str]:
        return list(self.network.get("nodes", {}).keys())

    def get_ports(self, node: str) -> List[str]:
        return list(
            self.network
            .get("nodes", {})
            .get(node, {})
            .get("ports", {})
            .keys()
        )

    def get_port_schedule(self, node: str, port: str) -> List[Dict[str, Any]]:
        return self.schedule.get(node, {}).get(port, [])

    def get_flow(self, flow_id: str) -> Dict[str, Any]:
        return self.flows.get(flow_id)

    def get_flow_path(self, flow_id: str) -> List[Dict[str, str]]:
        flow = self.get_flow(flow_id)
        return flow.get("path", []) if flow else []

    # ----------------------------
    # State inspection (debugging)
    # ----------------------------

    def pretty_print(self):
        print("=== Network Schedule State ===")
        for node, ports in self.schedule.items():
            print(f"Node: {node}")
            for port, entries in ports.items():
                print(f"  Port: {port}")
                for e in entries:
                    t = e.get("time", {})
                    print(
                        f"    Flow {e.get('flow_id')} "
                        f"[{t.get('start_us')} → {t.get('end_us')}]"
                    )

    # ----------------------------
    # Internal validation
    # ----------------------------

    def _validate_basic_structure(self):
        if "nodes" not in self.network:
            raise ValueError("NetworkState: 'network.nodes' missing")

        if "links" not in self.network:
            raise ValueError("NetworkState: 'network.links' missing")

        if not isinstance(self.flows, dict):
            raise ValueError("NetworkState: 'flows' must be a dict")

        if not isinstance(self.schedule, dict):
            raise ValueError("NetworkState: 'schedule' must be a dict")

    # ----------------------------
    # Validate references
    # ----------------------------
    
    def validate_references(self):
        nodes = self.network["nodes"]
        print(">>> validate_references CALLED")
        for node, ports in self.schedule.items():
            if node not in nodes:
                raise ValueError(f"Schedule references unknown node '{node}'")

            for port, entries in ports.items():
                if port not in nodes[node]["ports"]:
                    raise ValueError(
                        f"Schedule references unknown port '{port}' on node '{node}'"
                    )

                for e in entries:
                    flow_id = e.get("flow_id")
                    if flow_id not in self.flows:
                        raise ValueError(
                            f"Scheduled flow '{flow_id}' not found in flows section"
                        )

                    t = e.get("time", {})
                    if t.get("start_us") >= t.get("end_us"):
                        raise ValueError(
                            f"Invalid time window for flow '{flow_id}' on {node}:{port}"
                        )



    def _build_ports_from_links(self):
        """
        Builds implicit ports from links.
        Each link creates a bidirectional port.
        """
        for node in self.nodes.values():
            node.setdefault("ports", {})

        for link in self.links:
            src = link["src"]
            dst = link["dst"]
            cap = link.get("capacity_bps")

            self.nodes[src]["ports"][dst] = {"capacity_bps": cap}
            self.nodes[dst]["ports"][src] = {"capacity_bps": cap}