
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
from datetime import datetime
import logging
from collections import deque
import math


sys.path.append(os.path.join(os.getcwd(), "src"))
from src.network.Old_net import Network, FlowGenerator, PERIOD_SET
from src.env.Old_env import NetEnv  

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# -------------------------
# Rule-Based Controller
# -------------------------

class FlexTASRuleController:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.decision_history = []
        
    def get_action_and_confidence(self, observation):
        flow_feat = observation.get("flow_feature", np.zeros(20))
        
        # Extract features from NetEnv state structure
        if len(flow_feat) >= 7:
            period_ratio = flow_feat[-7] if len(flow_feat) >= 7 else 1.0
            jitter_ratio = flow_feat[-4] if len(flow_feat) >= 4 else 0.2
            accum_jitter_ratio = flow_feat[-2] if len(flow_feat) >= 2 else 0.0
        else:
            period_ratio = 0.5
            jitter_ratio = 0.3
            accum_jitter_ratio = 0.0
        
        # Rule-based scoring for TSN gating decisions
        score = 0.0
        
        # Rule 1: Strict jitter requirements need gating
        if jitter_ratio <= 0.2:
            score += 0.5
        
        # Rule 2: High-frequency streams (small period ratio) need gating
        if period_ratio <= 0.5:
            score += 0.3
        
        # Rule 3: High accumulated jitter needs gating to reset
        if accum_jitter_ratio > 0.5:
            score += 0.3
        
        confidence = min(score, 1.0)
        action = 1 if confidence >= 0.5 else 0
        
        self.decision_history.append({
            'action': action,
            'confidence': confidence,
            'jitter_ratio': jitter_ratio,
            'period_ratio': period_ratio,
            'accum_jitter_ratio': accum_jitter_ratio
        })
        
        return action, confidence


# -------------------------
# Progressive Independence Training Environment
# -------------------------

class ProgressiveIndependenceNetEnv(NetEnv):
    """
    Combines flow count curriculum (like TrainingNetEnv) with 
    rule-based progressive independence training
    """
    
    def __init__(self, graph, flow_generator, flow_counts=[10, 30, 50, 100, 150]):
        
        # Initialize flow curriculum
        self.flow_generator = flow_generator
        self.flow_counts = flow_counts
        self.current_flow_level = 0
        self.flows_changing_freq = 20  # Change flows after 20 successful episodes
        
        # Start with first flow count
        initial_flows = flow_generator(flow_counts[0])
        super().__init__(Network(graph, initial_flows))
        
        # Rule-based controller
        self.rule_controller = FlexTASRuleController()
        
        # Progressive independence phases
        self.phases = [
            {"name": "Bootstrap", "rule_strength": 0.5, "episodes": 50, "min_success": 0.6},
            {"name": "Mixed", "rule_strength": 0.3, "episodes": 80, "min_success": 0.7},
            {"name": "Reduced", "rule_strength": 0.15, "episodes": 100, "min_success": 0.7},
            {"name": "Minimal", "rule_strength": 0.05, "episodes": 120, "min_success": 0.75},
            {"name": "Independent", "rule_strength": 0.0, "episodes": float('inf'), "min_success": 0.0}
        ]
        
        self.current_phase = 0
        self.phase_episodes = 0
        self.episode_count = 0
        
        # Performance tracking
        self.phase_results = []
        self.phase_success_rate = 0.0
        self.flows_passed_count = 0
        
        # Rule intervention tracking
        self.rule_interventions = 0
        self.total_decisions = 0
        self.independence_test_freq = 10  # Force pure RL every 10 episodes
        
        # Logging
        self.logger = logging.getLogger(f"ProgressiveIndependence_{os.getpid()}")
        self.logger.setLevel(logging.INFO)
        
        # Current rule strength
        self.current_rule_strength = self.phases[0]["rule_strength"]
        
        self.logger.info(f" Starting Progressive Independence Training")
        self.logger.info(f"Flow counts: {flow_counts}")
        self.logger.info(f"Initial flows: {flow_counts[0]}")
        self.logger.info(f"Phase 0 ({self.phases[0]['name']}): Rule strength = {self.current_rule_strength}")
    
    def reset(self, **kwargs):
        """Reset with potential flow count changes"""
        # Check if we need to advance flow count
        if (self.flows_passed_count >= self.flows_changing_freq and 
            self.current_flow_level < len(self.flow_counts) - 1):
            
            self.current_flow_level += 1
            new_flow_count = self.flow_counts[self.current_flow_level]
            
            # Generate new flows
            new_flows = self.flow_generator(new_flow_count)
            
            # Reinitialize with new flows
            super().__init__(Network(self.graph, new_flows))
            
            self.flows_passed_count = 0
            self.logger.info(f" Advanced to {new_flow_count} flows (level {self.current_flow_level})")
        
        return super().reset(**kwargs)
    
    def step(self, action):
        """Step with progressive rule guidance"""
        self.episode_count += 1
        self.total_decisions += 1
        
        # Check if this is an independence test episode
        independence_test = (self.episode_count % self.independence_test_freq == 0)
        
        if independence_test:
            # Force pure RL (no rule guidance)
            guided_action = action
            rule_used = False
            self.logger.debug(f"Independence test episode {self.episode_count}")
        else:
            # Apply rule guidance based on current phase
            guided_action = self._apply_rule_guidance(action)
            rule_used = (guided_action != action)
        
        # Execute step
        obs, reward, done, trunc, info = super().step(guided_action)
        
        # Update phase progress when episode ends
        if done:
            self._update_phase_progress(info.get('success', False), independence_test)
        
        # Add curriculum info
        info.update({
            'phase': self.phases[self.current_phase]['name'],
            'rule_strength': self.current_rule_strength,
            'flow_level': self.current_flow_level,
            'flow_count': self.flow_counts[self.current_flow_level],
            'rule_used': rule_used,
            'independence_test': independence_test,
            'original_action': action,
            'guided_action': guided_action
        })
        
        return obs, reward, done, trunc, info
    
    def _apply_rule_guidance(self, rl_action):
        """Apply rule guidance based on current phase strength"""
        if self.current_rule_strength == 0.0:
            return rl_action  # Pure RL mode
        
        # Get rule recommendation
        state = self._generate_state()
        rule_action, rule_confidence = self.rule_controller.get_action_and_confidence(state)
        
        # Use rule guidance probabilistically
        if np.random.rand() < self.current_rule_strength:
            if rule_confidence > 0.6:  # Only use confident rules
                self.rule_interventions += 1
                self.logger.debug(f"Rule guidance: {rl_action} -> {rule_action} (conf: {rule_confidence:.3f})")
                return rule_action
        
        return rl_action
    
    def _update_phase_progress(self, success, independence_test):
        """Update phase progress and handle transitions"""
        self.phase_episodes += 1
        self.phase_results.append(success)
        
        # Track flow curriculum progress
        if success:
            self.flows_passed_count += 1
        
        # Calculate recent success rate (last 20 episodes)
        recent_window = 20
        if len(self.phase_results) >= recent_window:
            self.phase_success_rate = np.mean(self.phase_results[-recent_window:])
        else:
            self.phase_success_rate = np.mean(self.phase_results)
        
        current_phase_info = self.phases[self.current_phase]
        
        # Log progress
        if self.phase_episodes % 10 == 0:
            self.logger.info(
                f"Phase {self.current_phase} ({current_phase_info['name']}) - "
                f"Episode {self.phase_episodes}/{current_phase_info['episodes']} | "
                f"Success rate: {self.phase_success_rate:.3f} | "
                f"Flows: {self.flow_counts[self.current_flow_level]} | "
                f"Rule strength: {self.current_rule_strength:.3f}"
            )
        
        # Check phase transition conditions
        if len(self.phase_results) >= 15:  # Need some history
            ready_to_advance = (
                self.phase_episodes >= current_phase_info["episodes"] and
                self.phase_success_rate >= current_phase_info["min_success"] and
                self.current_phase < len(self.phases) - 1
            )
            
            if ready_to_advance:
                self._advance_phase()
            
            # Handle performance drops
            elif self.phase_success_rate < 0.3 and self.current_rule_strength < 0.2:
                # Temporarily increase rule strength if performance drops too much
                old_strength = self.current_rule_strength
                self.current_rule_strength = min(0.4, self.current_rule_strength + 0.1)
                self.logger.warning(
                    f" Performance drop! Increasing rule strength: "
                    f"{old_strength:.3f} -> {self.current_rule_strength:.3f}"
                )
    
    def _advance_phase(self):
        """Advance to next independence phase"""
        self.current_phase += 1
        self.phase_episodes = 0
        self.phase_results = []
        
        phase_info = self.phases[self.current_phase]
        self.current_rule_strength = phase_info["rule_strength"]
        
        self.logger.info(
            f" PHASE ADVANCE: {phase_info['name']} "
            f"(Rule strength: {self.current_rule_strength}) | "
            f"Previous success: {self.phase_success_rate:.3f}"
        )
    
    def get_curriculum_stats(self):
        """Get comprehensive curriculum statistics"""
        intervention_rate = (self.rule_interventions / self.total_decisions 
                           if self.total_decisions > 0 else 0.0)
        
        return {
            'current_phase': self.phases[self.current_phase]['name'],
            'phase_episodes': self.phase_episodes,
            'rule_strength': self.current_rule_strength,
            'success_rate': self.phase_success_rate,
            'flow_level': self.current_flow_level,
            'flow_count': self.flow_counts[self.current_flow_level],
            'intervention_rate': intervention_rate,
            'total_episodes': self.episode_count
        }


# -------------------------
# Training Callback
# -------------------------

class ProgressiveIndependenceCallback(BaseCallback):
    def __init__(self, env, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.log_freq == 0:
            stats = self.env.get_curriculum_stats()
            
            self.logger.record("curriculum/phase", stats['current_phase'])
            self.logger.record("curriculum/rule_strength", stats['rule_strength'])
            self.logger.record("curriculum/success_rate", stats['success_rate'])
            self.logger.record("curriculum/flow_count", stats['flow_count'])
            self.logger.record("curriculum/intervention_rate", stats['intervention_rate'])
            
            if self.verbose:
                print(f"Step {self.step_count}: {stats}")
        
        return True


# -------------------------
# Main Training Function
# -------------------------

def train_progressive_independence_flextas():
    """Main training function with Progressive Independence"""
    
    # Setup
    log_dir = "out/progressive_independence_training"
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ProgressiveTraining")
    
    # Load ERG graph
    graph_file = "graphs/ERG.graphml"
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    graph = nx.read_graphml(graph_file)
    logger.info(f"Loaded graph: {graph_file} with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Flow generator
    flow_gen = FlowGenerator(
        graph,
        seed=42,
        period_set=PERIOD_SET,
        jitters=[0.1, 0.2, 0.5]
    )
    
    # Create Progressive Independence Environment
    flow_counts = [10, 30, 50, 100, 150]
    env = ProgressiveIndependenceNetEnv(
        graph=graph,
        flow_generator=flow_gen,
        flow_counts=flow_counts
    )
    
    # Wrap in DummyVecEnv for stable-baselines3
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )
    
    # Setup callback
    callback = ProgressiveIndependenceCallback(env, log_freq=5000)
    
    # Training parameters
    total_timesteps = 500_000
    
    logger.info(f" Starting Progressive Independence Training")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Flow progression: {flow_counts}")
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="ProgressiveIndependence"
    )
    
    # Save model and results
    model.save(os.path.join(log_dir, "progressive_independence_model.zip"))
    
    # Save curriculum stats
    final_stats = env.get_curriculum_stats()
    with open(os.path.join(log_dir, "final_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    
    # Save rule controller
    with open(os.path.join(log_dir, "rule_controller.pkl"), "wb") as f:
        pickle.dump(env.rule_controller, f)
    
    logger.info(" Training completed!")
    logger.info(f"Final stats: {final_stats}")
    
    return model, env


# -------------------------
# Evaluation Function
# -------------------------

def evaluate_trained_model(model_path, graph_file="graphs/ERG.graphml", 
                          flow_counts=[10, 50, 100, 200], episodes_per_test=20):
    """Evaluate the trained model on different flow counts"""
    
    # Load graph
    graph = nx.read_graphml(graph_file)
    
    # Flow generator
    flow_gen = FlowGenerator(graph, seed=123, period_set=PERIOD_SET, jitters=[0.1, 0.2, 0.5])
    
    # Load model
    model = PPO.load(model_path)
    
    results = []
    
    for num_flows in flow_counts:
        print(f"\n=== Evaluating {num_flows} flows ===")
        
        flows = flow_gen(num_flows)
        env = NetEnv(Network(graph, flows))
        
        successes = 0
        
        for episode in range(episodes_per_test):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
            
            if info.get('success', False):
                successes += 1
        
        success_rate = successes / episodes_per_test
        results.append({
            'flow_count': num_flows,
            'success_rate': success_rate,
            'episodes': episodes_per_test
        })
        
        print(f"Flow count {num_flows}: {success_rate:.3f} success rate")
    
    return results


# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    # Train the model
    trained_model, training_env = train_progressive_independence_flextas()
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED!")
    print("="*60)
    
    # Show final curriculum stats
    final_stats = training_env.get_curriculum_stats()
    print(f" Final Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Evaluate on test cases
    print("\n Running evaluation...")
    model_path = "out/progressive_independence_training/progressive_independence_model.zip"
    eval_results = evaluate_trained_model(model_path)
    
    print("\n Evaluation Results:")
    for result in eval_results:
        print(f"  {result['flow_count']} flows: {result['success_rate']:.1%} success rate")