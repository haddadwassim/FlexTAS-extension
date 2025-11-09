
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
from typing import SupportsFloat, Any, Optional
from typing import Union, List, Optional, Dict, Any, Tuple

from definitions import ROOT_DIR, OUT_DIR, LOG_DIR
from src.lib.graph import neighbors_within_distance
from src.lib.operation import Operation, check_operation_isolation
from src.network.Old_net import Flow, Link, Net, PERIOD_SET, generate_cev, generate_flows, Network, _generate_graph, RandomGraph

MAX_NEIGHBORS = 20
MAX_REMAIN_HOPS = 10

class ErrorType(Enum):
    JitterExceed = auto()
    PeriodExceed = auto()
    GatingExceed = auto()

class SchedulingError(Exception):
    def __init__(self, error_type: ErrorType, msg):
        super().__init__(f"SchedulingError: {msg}")
        self.error_type: ErrorType = error_type
        self.msg: str = msg

class GlobalAwareStateEncoder:
    def __init__(self, env: 'NetEnv'):
        self.env = env
        self.max_neighbors = MAX_NEIGHBORS
        self.max_remain_hops = MAX_REMAIN_HOPS

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

        # pre-compute the neighbors for all links
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
            "remain_hops": spaces.Box(low=0, high=np.inf, shape=state['remain_hops'].shape, dtype=np.float32),
            "global_flow_features": spaces.Box(low=0, high=np.inf, shape=state['global_flow_features'].shape, dtype=np.float32),
            "scheduling_progress": spaces.Box(low=0, high=np.inf, shape=state['scheduling_progress'].shape, dtype=np.float32)
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
                hop_index,
                # Add current flow position in sorted order
                self.env.flow_index / len(self.env.flows)
            ]
        ], dtype=np.float32)
        return flow_feature

    def _neighbors_features(self, current_link):
        neighbors = self.neighbors_dict[current_link]

        if len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]  # Truncate to max_neighbors
        elif len(neighbors) < self.max_neighbors:
            neighbors += [-1] * (self.max_neighbors - len(neighbors))  # Pad with -1

        # Feature matrix and adjacency matrix handling
        feature_matrix = []
        edges = []
        max_edges = self.max_neighbors * (self.max_neighbors - 1)
        for idx, link_id in enumerate(neighbors):
            if link_id != -1:
                feature = self._link_feature(link_id)
            else:
                # Padding node: it must not be the first one.
                feature = np.zeros_like(feature_matrix[-1]) if feature_matrix else np.zeros(20, dtype=np.float32)

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

    def _get_global_flow_features(self):
        """Features about all flows in the network"""
        if not self.env.flows:
            return np.zeros(15, dtype=np.float32)
            
        flows = self.env.flows
        periods = [f.period for f in flows]
        jitters = [f.jitter / f.period for f in flows]
        path_lengths = [len(f.path) for f in flows]
        payloads = [f.payload for f in flows]
        
        return np.array([
            # Period statistics
            np.mean(periods) / Net.GCL_CYCLE_MAX,
            np.std(periods) / Net.GCL_CYCLE_MAX if len(periods) > 1 else 0,
            len(set(periods)) / len(periods),  # Period diversity
            
            # Jitter statistics  
            np.mean(jitters),
            np.std(jitters) if len(jitters) > 1 else 0,
            min(jitters) if jitters else 0,  # Most strict jitter requirement
            max(jitters) if jitters else 0,  # Most lenient jitter requirement
            
            # Path complexity
            np.mean(path_lengths) / 10.0,
            np.std(path_lengths) / 10.0 if len(path_lengths) > 1 else 0,
            max(path_lengths) / 10.0,  # Most complex path
            
            # Payload statistics
            np.mean(payloads) / Net.MTU,
            np.std(payloads) / Net.MTU if len(payloads) > 1 else 0,
            
            # Network-wide metrics
            len(flows) / 50.0,  # Normalized number of flows
            
            # Resource pressure (average GCL usage across all links)
            self._get_average_gcl_usage(),
            
            # Flow ordering context - where we are in the sorted order
            self.env.flow_index / len(flows) if flows else 0
        ], dtype=np.float32)
    
    def _get_scheduling_progress(self):
        """Progress and context about scheduling state"""
        if not self.env.flows:
            return np.zeros(8, dtype=np.float32)
            
        current_flow = self.env.current_flow()
        
        return np.array([
            # Overall progress
            self.env.flow_index / len(self.env.flows),  # Progress through flows
            
            # Per-flow progress
            len(self.env.temp_operations) / max(1, len(current_flow.path)),  # Hop progress
            
            # Resource utilization
            len(self.env.links_operations) / len(self.env.link_dict),  # Links with operations
            
            # Performance metrics
            self.env.reward / max(1, self.env.flow_index + 1),  # Average reward per flow
            
            # Current flow characteristics relative to all flows
            self._get_current_flow_ranking('period'),
            self._get_current_flow_ranking('jitter'),
            self._get_current_flow_ranking('path_length'),
            
            # Remaining challenge level
            self._estimate_remaining_difficulty()
        ], dtype=np.float32)

    def _get_average_gcl_usage(self):
        """Calculate average GCL usage across all links"""
        if not self.env.link_dict:
            return 0.0
            
        total_usage = 0.0
        for link in self.env.link_dict.values():
            gcl_info = self.env.links_gcl[link]
            if link.gcl_capacity > 0:
                usage = gcl_info.gcl_length / link.gcl_capacity
                total_usage += usage
        
        return total_usage / len(self.env.link_dict)

    def _get_current_flow_ranking(self, metric):
        """Get current flow's ranking in terms of a specific metric"""
        if not self.env.flows:
            return 0.0
            
        current_flow = self.env.current_flow()
        flows = self.env.flows
        
        if metric == 'period':
            values = [f.period for f in flows]
            current_value = current_flow.period
        elif metric == 'jitter':
            values = [f.jitter / f.period for f in flows]
            current_value = current_flow.jitter / current_flow.period
        elif metric == 'path_length':
            values = [len(f.path) for f in flows]
            current_value = len(current_flow.path)
        else:
            return 0.0
        
        # Return percentile ranking
        sorted_values = sorted(values)
        rank = sorted_values.index(current_value) if current_value in sorted_values else 0
        return rank / len(sorted_values)

    def _estimate_remaining_difficulty(self):
        """Estimate difficulty of remaining flows to schedule"""
        if self.env.flow_index >= len(self.env.flows):
            return 0.0
            
        remaining_flows = self.env.flows[self.env.flow_index:]
        if not remaining_flows:
            return 0.0
        
        # Difficulty factors: strict jitter requirements, long paths, large payloads
        difficulty_scores = []
        for flow in remaining_flows:
            jitter_strictness = 1.0 - (flow.jitter / flow.period)  # Higher = more strict
            path_complexity = len(flow.path) / 10.0
            payload_size = flow.payload / Net.MTU
            
            difficulty = (jitter_strictness * 0.5 + path_complexity * 0.3 + payload_size * 0.2)
            difficulty_scores.append(difficulty)
        
        return np.mean(difficulty_scores)

    def state(self):
        flow = self.env.flows[self.env.flow_index]
        current_link = self.env.current_link()

        flow_feature = self._flow_feature()
        link_feature = self._link_feature(current_link.link_id)
        edge_index, feature_matrix = self._neighbors_features(current_link.link_id)
        remain_hops_feature = self._remain_nodes_features(flow, current_link)
        global_flow_features = self._get_global_flow_features()
        scheduling_progress = self._get_scheduling_progress()

        return {
            "flow_feature": flow_feature,
            "link_feature": link_feature,
            "adjacency_matrix": edge_index,
            "features_matrix": feature_matrix,
            "remain_hops": remain_hops_feature,
            "global_flow_features": global_flow_features,
            "scheduling_progress": scheduling_progress
        }


class NetEnv(gym.Env):
    alpha: float = 1
    beta: float = 10
    gamma: float = 0.1

    @dataclass
    class GclInfo:
        gcl_cycle: int = 1
        gcl_length: int = 0

    def __init__(self, network: Network = None):
        super().__init__()

        if network is None:
            graph = generate_cev()
            network = Network(graph, generate_flows(graph, 10))

        self.graph = network.graph
        self.flows = network.flows
        self.line_graph, self.link_dict = network.line_graph, network.links_dict

        # Sort flows by period, jitter, and payload for better scheduling order
        self.flows = sorted(
            self.flows,
            key=lambda f: (f.period, f.jitter, f.payload)
        )

        assert self.graph is not None and self.flows is not None, "fail to init env, invalid graph or flows"

        self.num_flows: int = len(self.flows)

        self.links_operations: dict[Link, list[tuple[Flow, Operation]]] = defaultdict(list)
        self.temp_operations: list[tuple[Link, Operation]] = []
        self.links_gcl: dict[Link, NetEnv.GclInfo] = defaultdict(self._default_gcl_info)

        self.flow_index: int = 0
        self.last_action = None
        self.reward: float = 0

        self.state_encoder: GlobalAwareStateEncoder = GlobalAwareStateEncoder(self)
        self.observation_space: spaces.Dict = self.state_encoder.observation_space

        # Simple binary action space: enable gating or not for current operation
        self.action_space = spaces.Discrete(2)

        logger = logging.getLogger(f"{__name__}.{os.getpid()}")
        logger.setLevel(logging.INFO)
        self.logger = logger

    def _default_gcl_info(self):
        return self.GclInfo()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
            
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed)

        # Shuffle and re-sort flows for variety while maintaining good order
        random.shuffle(self.flows)
        self.flows = sorted(
            self.flows,
            key=lambda f: (f.period, f.jitter, f.payload)
        )

        self.links_operations.clear()
        self.temp_operations.clear()
        self.links_gcl.clear()

        self.flow_index = 0
        self.reward = 0

        return self._generate_state(), {}

    def _generate_state(self) -> ObsType:
        return self.state_encoder.state()

    def current_flow(self) -> Flow:
        return self.flows[self.flow_index]

    def current_link(self) -> Link:
        hop_index = len(self.temp_operations)
        flow = self.current_flow()
        link = self.link_dict[flow.path[hop_index]]
        return link

    def action_masks(self) -> np.ndarray:
        flow = self.current_flow()
        link = self.current_link()

        jitter_exceed = False
        if len(self.temp_operations) + 1 == len(flow.path) and len(self.temp_operations) != 0:
            # check accumulated jitter
            operation = self.temp_operations[-1][1]
            accum_jitter = operation.latest_time - operation.start_time \
                if operation.gating_time is None else 0

            if accum_jitter > flow.jitter:
                jitter_exceed = True

        can_gating = self.add_gating(link, flow.period, attempt=True)
        return np.array([not jitter_exceed, can_gating])

    def _check_temp_operations(self) -> Optional[int]:
        """
        :return: None if valid, else return the conflict operation
        """
        for link, operation in self.temp_operations:
            offset = self._check_valid_link(link, operation)
            if isinstance(offset, int):
                return offset
        return None

    def _check_valid_link(self, link: Link, operation: Operation) -> Optional[int]:
        # only needs to check whether the newly added operation is conflict with other operations.
        flow = self.current_flow()
        for flow_rhs, operation_rhs in self.links_operations[link]:
            offset = check_operation_isolation(
                (operation, flow.period),
                (operation_rhs, flow_rhs.period)
            )
            if offset is not None:
                return offset
        return None

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        :param action:
        :return:
        tuple: A tuple containing the following elements:
            - observation (object): The new state of the environment after the action.
            - reward (float): The reward for the action.
            - done (bool): A flag indicating whether the game has ended. True means the game has ended.
            - truncated (bool): always False.
            - info (dict): A dictionary with extra diagnostic information.
                'success' key indicates whether the game has been successfully completed.
                True means success, False means failure.
                'msg' key contains information for debug
        """
        gating = (action == 1)

        flow = self.current_flow()

        try:
            hop_index = len(self.temp_operations)

            link = self.link_dict[flow.path[hop_index]]

            trans_time = link.transmission_time(flow.payload)

            # compute enqueue_time min and max
            if hop_index == 0:
                earliest_enqueue_time = 0
                latest_enqueue_time = 0
            else:
                last_link, last_operation = self.temp_operations[-1]

                is_gating_last_link = self.last_action
                if is_gating_last_link:
                    earliest_dequeue_time = last_operation.gating_time
                    latest_dequeue_time = last_operation.gating_time
                else:
                    earliest_dequeue_time = last_operation.earliest_time
                    latest_dequeue_time = last_operation.latest_time

                earliest_enqueue_time = (earliest_dequeue_time
                                         + trans_time
                                         + Net.DELAY_PROP
                                         - Net.SYNC_PRECISION
                                         + Net.DELAY_PROC_MIN)

                latest_enqueue_time = (latest_dequeue_time
                                       + trans_time
                                       + Net.DELAY_PROP
                                       + Net.SYNC_PRECISION
                                       + Net.DELAY_PROC_MAX)

            # construct operation
            if gating:
                wait_time = 0  # no-wait
            else:
                wait_time = link.interference_time()  # might wait

            latest_dequeue_time = latest_enqueue_time + wait_time

            if hop_index == len(flow.path) - 1:
                # reach the dst, check jitter constraint.
                if not gating:
                    # don't need to check if gating, since gating reset the jitter.
                    accumulated_jitter = latest_enqueue_time - earliest_enqueue_time
                    if accumulated_jitter > flow.jitter:
                        raise SchedulingError(ErrorType.JitterExceed,
                                              f"jitter constraint unsatisfied. {accumulated_jitter} > {flow.jitter}")

            end_time = latest_dequeue_time + trans_time

            if end_time > flow.period:
                raise SchedulingError(ErrorType.PeriodExceed,
                                      "injection time is too late")

            operation = Operation(
                earliest_enqueue_time,
                None,
                latest_dequeue_time,
                end_time
            )
            if gating:
                operation.gating_time = latest_dequeue_time  # always enable gating right after the latest enqueue time.

            self.temp_operations.append((link, operation))

            while True:
                offset = self._check_temp_operations()
                if offset is None:
                    # find a valid solution that satisfies timing constraint
                    break

                assert isinstance(offset, int)

                for link, operation in self.temp_operations:
                    operation.add(offset)
                    if operation.end_time > flow.period:
                        # cannot be scheduled
                        raise SchedulingError(ErrorType.PeriodExceed, "timing isolation constraint unsatisfied.")

            gcl_added = 0
            if gating:
                # check gating constraint
                try:
                    old_gcl = self.links_gcl[link].gcl_length
                    self.add_gating(link, flow.period)
                    new_gcl = self.links_gcl[link].gcl_length
                    gcl_added = new_gcl - old_gcl
                except RuntimeError:
                    raise SchedulingError(ErrorType.GatingExceed,
                                          "gating constraint unsatisfied.")

            # Reward calculation
            reward_gcl = 0 - self.alpha * gcl_added / link.gcl_capacity if link.gcl_capacity != 0 else 0
            reward_time = 0 - self.beta * wait_time / flow.e2e_delay
            self.reward = 1 + reward_gcl + reward_time

        except SchedulingError as e:
            self.logger.info(f"end of episode, reason: [{e.error_type}: {e.msg}]\tScheduled flows: {self.flow_index}")
            done = True
            return self.observation_space.sample(), self.reward, done, False, {'success': False, 'msg': e.__str__()}

        done = False
        # successfully scheduling a flow
        if len(flow.path) == hop_index + 1:
            # reach the dst, all temp operations are confirmed.
            for link, operation in self.temp_operations:
                self.links_operations[link].append((flow, operation))
            self.temp_operations = []

            self.flow_index += 1

            if self.flow_index % math.ceil(self.num_flows * 0.1) == 0:
                # give an extra reward when the agent schedule another set of flows.
                self.reward += self.gamma * ((self.flow_index / self.num_flows) ** 2)

            if self.flow_index == len(self.flows):
                # Avoid generating state when episode is done
                return (
                    self.observation_space.sample(),  # or return None, or a dummy obs
                    self.reward,
                    True,
                    False,
                    {'success': True, 'ScheduleRes': self.links_operations.copy()}
                )

        self.last_action = gating

        return self._generate_state(), self.reward, done, False, {'success': done}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        gating = self.last_action
        self.logger.debug(f"Action: {gating}, Reward: {self.reward}")
        return

    def close(self):
        return

    def add_gating(self, link: Link, period: int, attempt: bool = False):
        gcl_info = self.links_gcl[link]
        gcl_cycle = gcl_info.gcl_cycle
        gcl_length = gcl_info.gcl_length
        new_cycle = math.lcm(gcl_cycle, period)
        new_length = gcl_length * (new_cycle // gcl_cycle)
        new_length += ((new_cycle // period) * 2)
        if new_length > link.gcl_capacity:
            if attempt:
                return False
            else:
                raise RuntimeError("Gating constraint is not satisfied.")
        elif not attempt:
            gcl_info.gcl_cycle = new_cycle
            gcl_info.gcl_length = new_length
        return True


class TrainingNetEnv(NetEnv):
    """
    Use curriculum learning to help training.
    """

    def __init__(self, graph, flow_generator, num_flows,
                 initial_ratio=0.2, step_ratio=0.05, changing_freq=10):

        self.flow_generator = flow_generator
        self.num_flows_target = num_flows
        self.num_flows_step = math.ceil(num_flows * step_ratio)

        num_flows_initial = math.ceil(num_flows * initial_ratio)
        flows = flow_generator(num_flows_initial)

        super().__init__(Network(graph, flows))

        self.num_passed = 0
        self.changing_freq = changing_freq

        log_file = os.path.join(LOG_DIR, f"training_env_{os.getpid()}.txt")
        fh = logging.FileHandler(filename=log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
                
                # Update flows with new sorted order
                self.flows = sorted(flows, key=lambda f: (f.period, f.jitter, f.payload))
                self.num_flows = len(self.flows)
                
                # Reset environment state
                self.links_operations.clear()
                self.temp_operations.clear()
                self.links_gcl.clear()
                self.flow_index = 0
                self.reward = 0

                self.logger.info(f"Great! The agent has already learnt how to solve the problem. "
                                 f"Change the flows to train the agent. num_flows: {num_flows}")

                self.num_passed = 0

        return res
