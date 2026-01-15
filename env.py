from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
import logging
import math
import numpy as np
import os
import pandas as pd
import random
from typing import SupportsFloat, Any, Optional, List, Dict

from definitions import ROOT_DIR, OUT_DIR, LOG_DIR
from src.lib.graph import neighbors_within_distance
from src.lib.operation import Operation, check_operation_isolation
from src.network.net import Flow, Link, Net, PERIOD_SET, generate_cev, generate_flows, Network
from network_state import NetworkState

MAX_NEIGHBORS = 20
MAX_REMAIN_HOPS = 10


@dataclass
class Flow:
    flow_id: str
    path: List[Dict[str, str]]  # [{"node": ..., "port": ...}]
    payload: int                # bytes
    period: int                 # max injection period (us)
    jitter: int                 # allowed jitter (us)
    e2e_delay: int              # max end-to-end delay (us)
    qos: Dict[str, int] = None  # optional QoS dict


class ErrorType(Enum):
    JitterExceed = auto()
    PeriodExceed = auto()
    GatingExceed = auto()


class SchedulingError(Exception):
    def __init__(self, error_type: ErrorType, msg):
        super().__init__(f"SchedulingError: {msg}")
        self.error_type: ErrorType = error_type
        self.msg: str = msg


class _StateEncoder:
    def __init__(self, env: 'NetEnv'):
        self.env = env
        self.max_neighbors = MAX_NEIGHBORS
        self.max_remain_hops = MAX_NEIGHBORS

        flows = self.env.flows
        self.periods_list = PERIOD_SET
        self.periods_list.sort()
        self.periods_one_hot_dict = pd.get_dummies(self.periods_list)

        link_dict = self.env.link_dict

        # [link, [period, num_flows]]
        self.link_flow_period_dict: dict = defaultdict(Counter)
        self.link_num_flows = defaultdict(int)
        for flow in flows:
            path = flow.path
            for link_id in path:
                link = link_dict[link_id]
                self.link_num_flows[link] += 1
                self.link_flow_period_dict[link][flow.period] += 1

        # pre-compute the neighbors for all links, to avoid heavy and duplicate computation during training
        self.neighbors_dict = {
            link: [link] + neighbors_within_distance(self.env.line_graph, link, 2)
            for link in link_dict
        }

        state = self.state()

        self.observation_space = spaces.Dict({
            "flow_feature": spaces.Box(low=0, high=np.inf, shape=state['flow_feature'].shape, dtype=np.float32),
            "link_feature": spaces.Box(low=0, high=np.inf, shape=state['link_feature'].shape, dtype=np.float32),
            "adjacency_matrix": spaces.Box(low=-1, high=self.max_neighbors,
                                           shape=state['adjacency_matrix'].shape,
                                           dtype=np.int64),
            "features_matrix": spaces.Box(low=0, high=np.inf, shape=state['features_matrix'].shape, dtype=np.float32),
            "remain_hops": spaces.Box(low=0, high=np.inf, shape=state['remain_hops'].shape, dtype=np.float32)
        })

    def _link_feature(self, link_id):
        link = self.env.link_dict[link_id]

        link_utilization = 0
        if len(self.env.links_operations[link]) != 0:
            for flow, operation in self.env.links_operations[link]:
                link_utilization += (operation.end_time - operation.start_time) / flow.period
        assert 0 <= link_utilization <= 1

        num_flows_to_schedule = self.link_num_flows[link] - len(self.env.links_operations[link])

        gcl_info = self.env.links_gcl[link]
        gcl_cycle = gcl_info.gcl_cycle
        gcl_length = gcl_info.gcl_length
        gcl_capacity = link.gcl_capacity

        link_gcl_feature = np.concatenate([
            self.periods_one_hot_dict[gcl_cycle]
            if gcl_cycle != 1 else np.zeros_like(self.periods_one_hot_dict.values[0]),
            [
                math.sqrt(link_utilization),  # do sqrt operation since this value is always quite small
                gcl_cycle / Net.GCL_CYCLE_MAX,
                gcl_length / gcl_capacity if gcl_capacity != 0 else 1,
                num_flows_to_schedule
            ]
        ], dtype=np.float32)

        if gcl_capacity != 0:
            link_flow_periods_feature = np.array([
                # num of flows of each period
                self.link_flow_period_dict[link][period] / gcl_capacity
                for period in self.periods_list
            ])
        else:
            link_flow_periods_feature = np.array([
                0 for _ in self.periods_list
            ])

        feature = np.concatenate((
            link_flow_periods_feature, link_gcl_feature
        ), dtype=np.float32)

        return feature

    def _flow_feature(self):
        flow = self.env.current_flow()

        accum_jitter = 0
        if len(self.env.temp_operations) != 0:
            operation = self.env.temp_operations[-1][1]
            accum_jitter = operation.latest_time - operation.start_time

        link = self.env.current_link()
        gcl_cycle = self.env.links_gcl[link].gcl_cycle

        hop_index = len(self.env.temp_operations)
        flow_feature = np.concatenate([
            self.periods_one_hot_dict[flow.period],
            [
                flow.period / Net.GCL_CYCLE_MAX,
                flow.period / gcl_cycle if gcl_cycle != 1 else 1,
                flow.payload / Net.MTU,
                flow.jitter / flow.period,
                flow.jitter / link.interference_time(),
                min(1, accum_jitter / flow.jitter) if flow.jitter != 0 else int(accum_jitter > 0),
                hop_index
            ]
        ], dtype=np.float32)
        return flow_feature

    def _neighbors_features(self, current_link):
        neighbors = self.neighbors_dict[current_link]

        if len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]  # Truncate to max_neighbors
        elif len(neighbors) < self.max_neighbors:
            neighbors += [-1] * (self.max_neighbors - len(neighbors))  # Pad with -1 or another invalid index

        # Feature matrix and adjacency matrix handling
        feature_matrix = []
        edges = []
        max_edges = self.max_neighbors * (self.max_neighbors - 1)
        for idx, link_id in enumerate(neighbors):
            if link_id != -1:
                feature = self._link_feature(link_id)
            else:
                # Padding node: it must not be the first one.
                feature = np.zeros_like(feature_matrix[-1])

            feature_matrix.append(feature)

            for jdx, dst_link in enumerate(neighbors):
                if dst_link == -1 or not self.env.line_graph.has_edge(link_id, dst_link):
                    continue
                edges.append([idx, jdx])

        if len(edges) < max_edges:
            # Pad edge_index to ensure consistent shape
            padded_edges = edges + [[-1, -1]] * (max_edges - len(edges))  # Pad with non-existent edge
        else:
            padded_edges = edges[:max_edges]  # Ensure it does not exceed max_edges

        edge_index = np.array(padded_edges, dtype=np.int64).T
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        return edge_index, feature_matrix

    def _remain_nodes_features(self, flow, current_link):
        path = flow.path
        features = []
        current_hop = None
        for i, link in enumerate(path):
            if current_link.link_id == link:
                current_hop = i
                break
        assert current_hop is not None

        for i in range(current_hop, len(path)):
            features.append(self._link_feature(path[i]))

        # features must not be empty
        assert len(features) > 0

        # padding
        while len(features) < self.max_remain_hops:
            features.append(np.zeros_like(features[-1]))

        # truncate
        if len(features) > self.max_remain_hops:
            features = features[:self.max_remain_hops]

        # flatten the features
        features = np.array(features, dtype=np.float32).ravel()

        return features

    def state(self):
        flow = self.env.flows[self.env.flow_index]
        current_link = self.env.current_link()

        flow_feature = self._flow_feature()
        link_feature = self._link_feature(current_link.link_id)
        edge_index, feature_matrix = self._neighbors_features(current_link.link_id)
        remain_hops_feature = self._remain_nodes_features(flow, current_link)

        return {
            "flow_feature": flow_feature,
            "link_feature": link_feature,
            "adjacency_matrix": edge_index,
            "features_matrix": feature_matrix,
            "remain_hops": remain_hops_feature
        }

class NetEnv(gym.Env):
    alpha: float = 1
    beta: float = 10
    gamma: float = 0.1

    @dataclass
    class GclInfo:
        gcl_cycle: int = 1
        gcl_length: int = 0

    def __init__(self, network=None):
        super().__init__()

        self.flows: List[Flow] = []
        self.flow_index: int = 0
        self.links_operations: dict = defaultdict(list)
        self.temp_operations: list = []
        self.links_gcl: dict = defaultdict(self._default_gcl_info)

        self.logger = logging.getLogger(f"{__name__}.{os.getpid()}")
        self.logger.setLevel(logging.INFO)

    def _default_gcl_info(self):
        return self.GclInfo()

    # -------------------------------
    # add_flow
    # -------------------------------
    def add_flow(self, flow: Flow, network_state: NetworkState) -> bool:
        """
        Schedule a single flow using FlexTAS logic and update the NetworkState.
        """
        self.flows.append(flow)
        self.flow_index = len(self.flows) - 1
        self.temp_operations = []

        done = False
        while not done:
            try:
                # Currently, we use a default action=1 (enable gating)
                obs, reward, done, truncated, info = self.step(action=1)
            except NotImplementedError:
                # -------------------------------
                # Baseline fallback for testing
                # -------------------------------
                print(f"[WARNING] Step not implemented, using baseline dummy schedule for {flow.flow_id}")
                # Fill NetworkState.schedule manually
                for hop in flow.path:
                    node, port = hop["node"], hop["port"]
                    if node not in network_state.schedule:
                        network_state.schedule[node] = {}
                    if port not in network_state.schedule[node]:
                        network_state.schedule[node][port] = []
                    network_state.schedule[node][port].append({
                        "flow_id": flow.flow_id,
                        "time": {"start_us": 0, "end_us": flow.period}
                    })
                done = True
                break

            if info.get("success") is False:
                print(f"[ERROR] Flow {flow.flow_id} could not be scheduled: {info.get('msg')}")
                return False

        # -------------------------------
        # Map scheduled operations to NetworkState (if step() works)
        # -------------------------------
        for link, operations in self.links_operations.items():
            for f, op in operations:
                if f.flow_id != flow.flow_id:
                    continue

                node, port = getattr(link, "src_node", None), getattr(link, "src_port", None)
                if node is None or port is None:
                    continue  # skip links without node/port info

                if node not in network_state.schedule:
                    network_state.schedule[node] = {}
                if port not in network_state.schedule[node]:
                    network_state.schedule[node][port] = []

                network_state.schedule[node][port].append({
                    "flow_id": f.flow_id,
                    "time": {
                        "start_us": getattr(op, "earliest_enqueue_time", 0),
                        "end_us": getattr(op, "end_time", flow.period)
                    }
                })

        print(f"[INFO] Flow {flow.flow_id} scheduled with FlexTAS logic.")
        return True

    # -------------------------------
    # schedule_flows
    # -------------------------------
    def schedule_flows(self, flows: List[Flow], network_state: NetworkState):
        """
        Schedule multiple flows sequentially using add_flow().
        """
        for flow in flows:
            success = self.add_flow(flow, network_state)
            if not success:
                print(f"[WARNING] Failed to schedule flow {flow.flow_id}")





class TrainingNetEnv(NetEnv):
    """
    Use curriculum learning to help training.
    Begin with an easy environment that contains a small set of flows for agent to learn,
    increase the number of flows gradually to make the env harder if the agent can easily
    pass the current env.
    Once the agent has learnt how to schedule current flows, generate a new flow set for training.
    """

    def __init__(self, graph, flow_generator, num_flows,
                 initial_ratio=0.2, step_ratio=0.05, changing_freq=10):

        self.flow_generator = flow_generator

        self.num_flows_target = num_flows

        # the number of flows newly added each time changing the env.
        self.num_flows_step = math.ceil(num_flows * step_ratio)

        # start with half of the target num_flows and incrementally add flows if agent has learnt to schedule.
        num_flows_initial = math.ceil(num_flows * initial_ratio)
        flows = flow_generator(num_flows_initial)

        super().__init__(Network(graph, flows))

        self.num_passed = 0
        self.changing_freq = changing_freq

        log_file = os.path.join(LOG_DIR, f"training_env_{os.getpid()}.txt")
        fh = logging.FileHandler(filename=log_file)
        fh.setLevel(logging.DEBUG)
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Add the formatter to the handler
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(f"Start training with {num_flows_initial} flows.")

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        res = super().step(action)

        done, info = res[2], res[-1]
        if done and info['success']:
            self.num_passed += 1
            self.logger.info(f"passed the job! ({self.num_passed})")

            if self.num_passed == self.changing_freq:
                num_flows = min(self.num_flows_target, self.num_flows + self.num_flows_step)
                flows = self.flow_generator(num_flows)
                super().__init__(Network(self.graph, flows))
                self.logger.info(f"Great! The agent has already learnt how to solve the problem. "
                                 f"Change the flows to train the agent. num_flows: {num_flows}")

                self.num_passed = 0

        return res
