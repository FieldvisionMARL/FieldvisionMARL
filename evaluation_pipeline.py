# -*- coding: utf-8 -*-
"""
FieldVision Evaluation Pipeline
===============================

Baseline policies, metric collection, visualization, and replication tools.

Usage:
    from fieldvision_complete import make_multi_drone_env, MAPPOWrapper, get_difficulty_config
    from evaluation_pipeline import *
    
    results = evaluate_all_baselines("streaming_hard", n_episodes=30)
    generate_all_figures(results, "figures/")
    save_checkpoint("checkpoint/", results)
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

# =============================================================================
# BASELINE POLICIES
# =============================================================================

class BasePolicy:
    name = "Base"
    def get_action(self, obs, env=None) -> np.ndarray:
        raise NotImplementedError
    def reset(self):
        pass

class LocalPolicy(BasePolicy):
    name = "Local-Only"
    def get_action(self, obs, env=None):
        return np.zeros(4, dtype=np.int64)

class EdgePolicy(BasePolicy):
    name = "Edge-Only"
    def get_action(self, obs, env=None):
        return np.ones(4, dtype=np.int64)

class CloudPolicy(BasePolicy):
    name = "Cloud-Only"
    def get_action(self, obs, env=None):
        return np.full(4, 2, dtype=np.int64)

class RoundRobinPolicy(BasePolicy):
    """Each drone independently cycles Local→Edge→Cloud, advancing only on successful actions."""
    name = "Round-Robin"
    def __init__(self):
        # Each drone has its own counter, staggered start
        self.counters = [0, 1, 2, 0]
        self.last_actions = [0, 0, 0, 0]
    def get_action(self, obs, env=None):
        actions = np.array([c % 3 for c in self.counters], dtype=np.int64)
        self.last_actions = actions.tolist()
        # Always advance counters (simple time-based cycling)
        self.counters = [c + 1 for c in self.counters]
        return actions
    def reset(self):
        self.counters = [0, 1, 2, 0]
        self.last_actions = [0, 0, 0, 0]


class RoundRobinSyncPolicy(BasePolicy):
    """All drones cycle together: all Local, then all Edge, then all Cloud."""
    name = "Round-Robin-Sync"
    def __init__(self):
        self.step = 0
    def get_action(self, obs, env=None):
        action = self.step % 3
        self.step += 1
        return np.array([action, action, action, action], dtype=np.int64)
    def reset(self):
        self.step = 0

class RandomPolicy(BasePolicy):
    name = "Random"
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
    def get_action(self, obs, env=None):
        return self.rng.integers(0, 3, size=4)

class MultiMetricPolicy(BasePolicy):
    """DCOSS-IoT inspired Multi-Metric Decision Engine.
    
    Tuned for agricultural drone offloading with:
    - Device-aware processing (MPD vs LPD capabilities)
    - Network-aware offloading (bandwidth, stability, congestion)
    - Deadline-aware urgency weighting
    
    Observation indices (16 features):
    - 0: queue_len/10, 1: deadline_ratio, 2: priority, 3: pending/10
    - 4: battery, 5: load, 6: active_tasks, 7: is_available
    - 8: speed/20 (MPD=1.0, LPD=0.25), 9: battery_drain*100 (MPD=0.5, LPD=1.0)
    - 10: bandwidth/15, 11: latency/200, 12: connected, 13: loss_rate
    - 14: GPU busy ratio, 15: Cloud busy ratio
    """
    def __init__(self, w_urgency=0.3, w_bandwidth=0.25, w_stability=0.2, 
                 w_device=0.15, w_congestion=0.1, name="Multi-Metric"):
        self.w_urgency = w_urgency
        self.w_bandwidth = w_bandwidth
        self.w_stability = w_stability
        self.w_device = w_device
        self.w_congestion = w_congestion
        self.name = name
        self.bw_history = defaultdict(list)
    
    def reset(self):
        self.bw_history = defaultdict(list)
    
    def get_action(self, obs, env=None):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        actions = []
        for i in range(min(obs.shape[0], 4)):
            o = obs[i]
            
            # Extract features (handle variable observation lengths)
            deadline_ratio = o[1] if len(o) > 1 else 0.5
            battery = o[4] if len(o) > 4 else 0.5
            speed = o[8] if len(o) > 8 else 0.5  # MPD=1.0, LPD=0.25
            bandwidth = o[10] if len(o) > 10 else 0.5  # normalized by /15
            connected = o[12] if len(o) > 12 else 1.0
            loss_rate = o[13] if len(o) > 13 else 0.0
            gpu_busy = o[14] if len(o) > 14 else 0.0
            cloud_busy = o[15] if len(o) > 15 else 0.0
            
            # Compute decision factors
            urgency = 1.0 - deadline_ratio  # Higher when deadline closer
            
            # Track bandwidth stability
            self.bw_history[i].append(bandwidth)
            if len(self.bw_history[i]) > 10:
                self.bw_history[i] = self.bw_history[i][-10:]
            stability = 1.0 - min(np.std(self.bw_history[i]) * 5, 1.0) if len(self.bw_history[i]) >= 3 else 0.5
            
            # Device capability score (high for MPD, low for LPD)
            device_local_capability = speed  # MPD=1.0 good at local, LPD=0.25 bad at local
            
            # Network quality
            network_quality = bandwidth * connected * (1.0 - loss_rate)
            
            # Compute scores for each action
            # Local: Good when device is capable (MPD), urgent, or network is poor
            local_score = (
                self.w_device * device_local_capability +
                self.w_urgency * urgency * device_local_capability +  # Urgency helps local only for MPDs
                self.w_bandwidth * (1.0 - network_quality) * 0.5 +  # Slight boost when network is bad
                self.w_congestion * (gpu_busy + cloud_busy) * 0.3  # Prefer local when offload targets busy
            )
            
            # Edge: Good when network is stable and GPU not busy
            edge_score = (
                self.w_bandwidth * network_quality +
                self.w_stability * stability +
                self.w_urgency * urgency * 0.5 +  # Some urgency benefit
                self.w_device * (1.0 - device_local_capability) +  # LPDs prefer edge
                self.w_congestion * (1.0 - gpu_busy)  # Prefer when GPU is free
            )
            
            # Cloud: Good when bandwidth is excellent and cloud not busy
            cloud_score = (
                self.w_bandwidth * max(0, network_quality - 0.5) * 2 +  # Only when BW > 0.5
                self.w_stability * stability * 0.8 +
                self.w_congestion * (1.0 - cloud_busy) +
                self.w_urgency * (1.0 - urgency) * 0.3  # Cloud better for non-urgent (latency)
            )
            
            # If not connected, force local
            if connected < 0.5:
                actions.append(0)
            else:
                scores = [local_score, edge_score, cloud_score]
                actions.append(np.argmax(scores))
        
        while len(actions) < 4:
            actions.append(0)
        return np.array(actions, dtype=np.int64)


class LatencyGreedyPolicy(BasePolicy):
    name = "Latency-Greedy"
    def get_action(self, obs, env=None):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        actions = []
        for i in range(min(obs.shape[0], 4)):
            o = obs[i]
            # Bandwidth at index 8 in new 12-feature observation
            bw = o[8] if len(o) > 8 else 0.5
            
            local_lat = 10.0
            edge_lat = 10.0 / (bw * 10 + 0.1) + 5.0
            cloud_lat = 10.0 / (bw * 10 + 0.1) * 1.5 + 2.0
            
            actions.append(np.argmin([local_lat, edge_lat, cloud_lat]))
        
        while len(actions) < 4:
            actions.append(0)
        return np.array(actions, dtype=np.int64)

def get_all_policies(seed=42) -> List[BasePolicy]:
    """Get all baseline policies for evaluation.
    
    Includes:
    - Simple fixed policies (Local, Edge, Cloud)
    - Cycling policies (Round-Robin, Random)
    - Adaptive heuristics (Multi-Metric variants, Latency-Greedy)
    """
    return [
        LocalPolicy(),
        EdgePolicy(),
        CloudPolicy(),
        RoundRobinPolicy(),
        RandomPolicy(seed),
        # Multi-Metric with different weight profiles
        MultiMetricPolicy(0.30, 0.25, 0.20, 0.15, 0.10, "Multi-Metric"),  # Balanced
        MultiMetricPolicy(0.20, 0.35, 0.25, 0.10, 0.10, "Multi-Metric-BW"),  # Bandwidth-focused
        MultiMetricPolicy(0.40, 0.15, 0.15, 0.20, 0.10, "Multi-Metric-Urgent"),  # Urgency-focused
        LatencyGreedyPolicy(),
    ]

# =============================================================================
# METRIC COLLECTION
# =============================================================================

@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    tasks_completed: int
    total_tasks: int
    completion_rate: float
    deadline_misses: int      # Mosaic-level deadline misses
    transfer_failures: int
    tasks_on_time: int        # Per-task: completed before individual deadline
    tasks_late: int           # Per-task: completed after individual deadline
    on_time_rate: float       # % of COMPLETED tasks that were on-time
    deadline_compliance: float # % of ALL tasks completed on-time (tasks_on_time / total_tasks)
    avg_latency: float        # Average task latency (timesteps)
    battery_used_pct: float   # Battery consumed (%)
    local_pct: float
    edge_pct: float
    cloud_pct: float
    avg_bandwidth: float

class MetricCollector:
    def __init__(self, name: str):
        self.name = name
        self.episodes: List[EpisodeMetrics] = []
        self._actions = []
        self._bandwidths = []
        self._task_latencies = []  # Track per-task latencies within episode
        self.all_task_latencies = []  # Store all task latencies across episodes for CDF
    
    def reset_episode(self):
        self._actions = []
        self._bandwidths = []
        self._task_latencies = []
    
    def step(self, actions, bandwidth=5.0, latency=None):
        self._actions.extend(actions.tolist())
        self._bandwidths.append(bandwidth)
        if latency is not None:
            self._task_latencies.append(latency)
    
    def record_task_latency(self, latency_timesteps: float):
        """Record latency for a completed task (in timesteps from creation to completion)."""
        self._task_latencies.append(latency_timesteps)
        self.all_task_latencies.append(latency_timesteps)
    
    def finish_episode(self, episode, reward, steps, sim):
        counts = {0: 0, 1: 0, 2: 0}
        for a in self._actions:
            counts[a] = counts.get(a, 0) + 1
        total = max(sum(counts.values()), 1)
        
        tasks_on_time = getattr(sim, 'tasks_on_time', 0)
        tasks_late = getattr(sim, 'tasks_late', 0)
        total_tasks = getattr(sim, 'total_chunks', 1)
        
        # on_time_rate: % of ALL tasks completed on-time (the meaningful metric)
        on_time_rate = tasks_on_time / max(total_tasks, 1) * 100
        
        # deadline_compliance is same as on_time_rate now (kept for compatibility)
        deadline_compliance = on_time_rate
        
        # Battery usage - get from sim's compute_agents
        battery_used_pct = 0.0
        if hasattr(sim, 'compute_agents'):
            drone_names = ['MPD_0', 'MPD_1', 'LPD_0', 'LPD_1']
            batteries = []
            for name in drone_names:
                if name in sim.compute_agents:
                    batteries.append(sim.compute_agents[name].battery_level)
            if batteries:
                battery_used_pct = (1.0 - np.mean(batteries)) * 100
        
        # Average latency from this episode
        avg_latency = np.mean(self._task_latencies) if self._task_latencies else 0.0
        
        m = EpisodeMetrics(
            episode=episode, total_reward=reward, steps=steps,
            tasks_completed=getattr(sim, 'processed_chunks_count', 0),
            total_tasks=total_tasks,
            completion_rate=getattr(sim, 'processed_chunks_count', 0) / max(total_tasks, 1),
            deadline_misses=getattr(sim, 'deadline_misses', 0),
            transfer_failures=getattr(sim, 'transfer_failures', 0),
            tasks_on_time=tasks_on_time,
            tasks_late=tasks_late,
            on_time_rate=on_time_rate,
            deadline_compliance=deadline_compliance,
            avg_latency=avg_latency,
            battery_used_pct=battery_used_pct,
            local_pct=counts[0] / total * 100,
            edge_pct=counts[1] / total * 100,
            cloud_pct=counts[2] / total * 100,
            avg_bandwidth=np.mean(self._bandwidths) if self._bandwidths else 0,
        )
        self.episodes.append(m)
        self.reset_episode()
    
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.episodes])
    
    def summary(self) -> Dict:
        df = self.to_df()
        return {
            "policy": self.name,
            "reward_mean": df["total_reward"].mean(),
            "reward_std": df["total_reward"].std(),
            "completion": df["completion_rate"].mean() * 100,
            "on_time_rate": df["on_time_rate"].mean(),
            "deadline_compliance": df["deadline_compliance"].mean(),
            "deadline_misses": df["deadline_misses"].mean(),
            "transfer_failures": df["transfer_failures"].mean(),
            "avg_latency": df["avg_latency"].mean(),
            "battery_used_pct": df["battery_used_pct"].mean(),
            "local_pct": df["local_pct"].mean(),
            "edge_pct": df["edge_pct"].mean(),
            "cloud_pct": df["cloud_pct"].mean(),
        }


class MARLResultsAdapter:
    """Adapter to make MARL results compatible with MetricCollector interface."""
    
    def __init__(self, name: str, stats: Dict):
        """
        Create adapter from MARL stats dict.
        
        Args:
            name: Algorithm name (e.g., 'mappo', 'mapo')
            stats: Dict with keys: reward_mean, reward_std, rewards (list), 
                   completion, completions (list), deadline_misses, deadline_misses_list,
                   transfer_failures, transfer_failures_list, on_time_rate, on_time_rates_list,
                   avg_latency, battery_used_pct, local_pct, edge_pct, cloud_pct
        """
        self.name = name
        self._stats = stats
        self.all_task_latencies = []  # MARL doesn't track this currently
        
        # Create episodes from per-episode data
        self.episodes = []
        rewards = stats.get('rewards', [])
        completions = stats.get('completions', [])
        deadline_misses = stats.get('deadline_misses_list', [])
        transfer_failures = stats.get('transfer_failures_list', [])
        on_time_rates = stats.get('on_time_rates_list', [])
        tasks_on_time_list = stats.get('tasks_on_time_list', [])
        tasks_late_list = stats.get('tasks_late_list', [])
        ep_latencies = stats.get('ep_mean_latencies', [])
        
        for i, r in enumerate(rewards):
            # Use per-episode values if available, else use aggregate
            comp = completions[i] / 100.0 if i < len(completions) else stats.get('completion', 0) / 100.0
            dm = deadline_misses[i] if i < len(deadline_misses) else stats.get('deadline_misses', 0)
            tf = transfer_failures[i] if i < len(transfer_failures) else stats.get('transfer_failures', 0)
            otr = on_time_rates[i] if i < len(on_time_rates) else stats.get('on_time_rate', 100.0)
            tot = tasks_on_time_list[i] if i < len(tasks_on_time_list) else 0
            tl = tasks_late_list[i] if i < len(tasks_late_list) else 0
            lat = ep_latencies[i] if i < len(ep_latencies) else stats.get('avg_latency', 0)
            
            # Calculate deadline_compliance from tasks_on_time / total_tasks
            # Estimate total_tasks from completion rate
            total_tasks_est = int(tot / max(comp, 0.01)) if comp > 0 else 48
            deadline_compliance = tot / max(total_tasks_est, 1) * 100
            
            self.episodes.append(EpisodeMetrics(
                episode=i,
                total_reward=float(r),
                steps=0,
                tasks_completed=int(tot + tl),
                total_tasks=total_tasks_est,
                completion_rate=comp,
                deadline_misses=int(dm),
                transfer_failures=int(tf),
                tasks_on_time=int(tot),
                tasks_late=int(tl),
                on_time_rate=float(otr),
                deadline_compliance=deadline_compliance,
                avg_latency=float(lat),
                battery_used_pct=stats.get('battery_used_pct', 0),
                local_pct=stats.get('local_pct', 0),
                edge_pct=stats.get('edge_pct', 0),
                cloud_pct=stats.get('cloud_pct', 0),
                avg_bandwidth=0,
            ))
    
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.episodes])
    
    def summary(self) -> Dict:
        return {
            "policy": self.name,
            "reward_mean": self._stats.get('reward_mean', 0),
            "reward_std": self._stats.get('reward_std', 0),
            "completion": self._stats.get('completion', 0),
            "on_time_rate": self._stats.get('on_time_rate', 100.0),
            "deadline_compliance": self._stats.get('completion', 0),  # Approximation
            "deadline_misses": self._stats.get('deadline_misses', 0),
            "transfer_failures": self._stats.get('transfer_failures', 0),
            "avg_latency": self._stats.get('avg_latency', 0),
            "battery_used_pct": self._stats.get('battery_used_pct', 0),
            "local_pct": self._stats.get('local_pct', 0),
            "edge_pct": self._stats.get('edge_pct', 0),
            "cloud_pct": self._stats.get('cloud_pct', 0),
        }


def convert_marl_results(marl_results: Dict) -> Dict[str, MARLResultsAdapter]:
    """Convert MARL results dict to adapter objects for consistent handling."""
    adapted = {}
    for alg, data in marl_results.items():
        if isinstance(data, dict):
            # New format: dict with reward_mean, reward_std, rewards, etc.
            adapted[alg] = MARLResultsAdapter(alg, data)
        elif isinstance(data, list):
            # Old format: list of rewards
            adapted[alg] = MARLResultsAdapter(alg, {
                'reward_mean': np.mean(data),
                'reward_std': np.std(data),
                'rewards': data,
                'completion': 0,
                'local_pct': 0,
                'edge_pct': 0,
                'cloud_pct': 0,
            })
        elif isinstance(data, MetricCollector):
            adapted[alg] = data
        elif isinstance(data, MARLResultsAdapter):
            adapted[alg] = data
    return adapted


# =============================================================================
# COMPREHENSIVE METRIC COLLECTION (Per-Step Data)
# =============================================================================

@dataclass
class StepMetrics:
    """Per-step detailed metrics for deep analysis."""
    episode: int
    step: int
    # Actions per agent
    action_mpd0: int
    action_mpd1: int
    action_lpd0: int
    action_lpd1: int
    # Rewards per agent
    reward_mpd0: float
    reward_mpd1: float
    reward_lpd0: float
    reward_lpd1: float
    reward_total: float
    # Network state per agent
    bw_mpd0: float
    bw_mpd1: float
    bw_lpd0: float
    bw_lpd1: float
    # Queue state
    queue_mpd0: int
    queue_mpd1: int
    queue_lpd0: int
    queue_lpd1: int
    # Battery levels per agent
    battery_mpd0: float
    battery_mpd1: float
    battery_lpd0: float
    battery_lpd1: float
    # Resource utilization
    gpu_busy: bool
    cloud_busy: bool
    # Timestep
    timestep: int


class ComprehensiveMetricCollector:
    """Collects detailed per-step metrics for deep analysis."""
    
    def __init__(self, name: str, scenario: str = "unknown"):
        self.name = name
        self.scenario = scenario
        self.step_metrics: List[StepMetrics] = []
        self.episodes: List[EpisodeMetrics] = []
        self._current_episode = 0
        self._current_step = 0
        self._episode_reward = 0.0
        self._episode_actions = []
        self._bandwidths = []
        self.all_task_latencies = []
    
    def reset_episode(self):
        self._current_step = 0
        self._episode_reward = 0.0
        self._episode_actions = []
        self._bandwidths = []
    
    def record_step(self, actions: np.ndarray, rewards: np.ndarray, sim, timestep: int):
        """Record comprehensive per-step data."""
        # Get network states
        bws = [0.0, 0.0, 0.0, 0.0]
        queues = [0, 0, 0, 0]
        batteries = [1.0, 1.0, 1.0, 1.0]
        agent_names = ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]
        
        if hasattr(sim, 'drone_network_states'):
            for i, name in enumerate(agent_names):
                if name in sim.drone_network_states:
                    bws[i] = sim.drone_network_states[name].drone_ground_bandwidth_mbps
        
        if hasattr(sim, 'task_queues'):
            for i, name in enumerate(agent_names):
                if name in sim.task_queues:
                    queues[i] = len(sim.task_queues[name])
        
        # Get battery levels
        if hasattr(sim, 'compute_agents'):
            for i, name in enumerate(agent_names):
                if name in sim.compute_agents:
                    batteries[i] = getattr(sim.compute_agents[name], 'battery_level', 1.0)
        
        # Get resource states
        gpu_busy = False
        cloud_busy = False
        if hasattr(sim, 'compute_agents'):
            gpu = sim.compute_agents.get("Desktop GPU")
            cloud = sim.compute_agents.get("Cloud")
            if gpu:
                gpu_busy = getattr(gpu, 'busy', False)
            if cloud:
                cloud_busy = getattr(cloud, 'busy', False)
        
        # Ensure rewards is array
        if isinstance(rewards, (int, float)):
            rewards = np.array([rewards / 4] * 4)
        elif isinstance(rewards, dict):
            rewards = np.array([rewards.get(n, 0) for n in agent_names])
        
        step_data = StepMetrics(
            episode=self._current_episode,
            step=self._current_step,
            action_mpd0=int(actions[0]),
            action_mpd1=int(actions[1]),
            action_lpd0=int(actions[2]),
            action_lpd1=int(actions[3]),
            reward_mpd0=float(rewards[0]),
            reward_mpd1=float(rewards[1]),
            reward_lpd0=float(rewards[2]),
            reward_lpd1=float(rewards[3]),
            reward_total=float(rewards.sum()),
            bw_mpd0=bws[0],
            bw_mpd1=bws[1],
            bw_lpd0=bws[2],
            bw_lpd1=bws[3],
            queue_mpd0=queues[0],
            queue_mpd1=queues[1],
            queue_lpd0=queues[2],
            queue_lpd1=queues[3],
            battery_mpd0=batteries[0],
            battery_mpd1=batteries[1],
            battery_lpd0=batteries[2],
            battery_lpd1=batteries[3],
            gpu_busy=gpu_busy,
            cloud_busy=cloud_busy,
            timestep=timestep,
        )
        self.step_metrics.append(step_data)
        
        self._episode_reward += float(rewards.sum())
        self._episode_actions.extend(actions.tolist())
        self._bandwidths.append(np.mean(bws))
        self._current_step += 1
    
    def record_task_latency(self, latency: float):
        self.all_task_latencies.append(latency)
    
    def finish_episode(self, sim):
        """Finalize episode metrics."""
        counts = {0: 0, 1: 0, 2: 0}
        for a in self._episode_actions:
            counts[a] = counts.get(a, 0) + 1
        total = max(sum(counts.values()), 1)
        
        tasks_on_time = getattr(sim, 'tasks_on_time', 0)
        tasks_late = getattr(sim, 'tasks_late', 0)
        total_tasks = getattr(sim, 'total_chunks', 1)
        
        # on_time_rate: % of ALL tasks completed on-time
        on_time_rate = tasks_on_time / max(total_tasks, 1) * 100
        
        # Battery usage
        battery_used_pct = 0.0
        if hasattr(sim, 'compute_agents'):
            drone_names = ['MPD_0', 'MPD_1', 'LPD_0', 'LPD_1']
            batteries = []
            for name in drone_names:
                if name in sim.compute_agents:
                    batteries.append(sim.compute_agents[name].battery_level)
            if batteries:
                battery_used_pct = (1.0 - np.mean(batteries)) * 100
        
        m = EpisodeMetrics(
            episode=self._current_episode,
            total_reward=self._episode_reward,
            steps=self._current_step,
            tasks_completed=getattr(sim, 'processed_chunks_count', 0),
            total_tasks=total_tasks,
            completion_rate=getattr(sim, 'processed_chunks_count', 0) / max(total_tasks, 1),
            deadline_misses=getattr(sim, 'deadline_misses', 0),
            transfer_failures=getattr(sim, 'transfer_failures', 0),
            tasks_on_time=tasks_on_time,
            tasks_late=tasks_late,
            on_time_rate=on_time_rate,
            deadline_compliance=on_time_rate,
            avg_latency=0.0,
            battery_used_pct=battery_used_pct,
            local_pct=counts[0] / total * 100,
            edge_pct=counts[1] / total * 100,
            cloud_pct=counts[2] / total * 100,
            avg_bandwidth=np.mean(self._bandwidths) if self._bandwidths else 0,
        )
        self.episodes.append(m)
        self._current_episode += 1
        self.reset_episode()
    
    def to_df(self) -> pd.DataFrame:
        """Episode-level dataframe (compatible with MetricCollector)."""
        return pd.DataFrame([asdict(e) for e in self.episodes])
    
    def steps_to_df(self) -> pd.DataFrame:
        """Step-level detailed dataframe for deep analysis."""
        return pd.DataFrame([asdict(s) for s in self.step_metrics])
    
    def summary(self) -> Dict:
        df = self.to_df()
        return {
            "policy": self.name,
            "scenario": self.scenario,
            "reward_mean": df["total_reward"].mean(),
            "reward_std": df["total_reward"].std(),
            "completion": df["completion_rate"].mean() * 100,
            "on_time_rate": df["on_time_rate"].mean(),
            "deadline_compliance": df["deadline_compliance"].mean(),
            "deadline_misses": df["deadline_misses"].mean(),
            "transfer_failures": df["transfer_failures"].mean(),
            "avg_latency": df["avg_latency"].mean(),
            "battery_used_pct": df["battery_used_pct"].mean(),
            "local_pct": df["local_pct"].mean(),
            "edge_pct": df["edge_pct"].mean(),
            "cloud_pct": df["cloud_pct"].mean(),
            "n_episodes": len(self.episodes),
            "n_steps": len(self.step_metrics),
        }


def evaluate_policy_comprehensive(policy, make_env_fn, scenario: str, 
                                   n_episodes=30, max_steps=300, seed=42) -> ComprehensiveMetricCollector:
    """Evaluate policy with comprehensive per-step metrics."""
    collector = ComprehensiveMetricCollector(policy.name, scenario)
    
    for ep in range(n_episodes):
        env = make_env_fn(seed + ep)
        
        reset_result = env.reset()
        # Handle both tuple returns (obs, info) and dict returns
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        
        if isinstance(obs, dict):
            obs = np.stack([obs[n] for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
        
        policy.reset()
        collector.reset_episode()
        
        sim = env.sim if hasattr(env, 'sim') else (env.env.sim if hasattr(env, 'env') else env)
        
        for step in range(max_steps):
            actions = policy.get_action(obs, sim)
            
            # Convert actions to dict format for env.step
            action_dict = {n: int(actions[i]) for i, n in enumerate(["MPD_0", "MPD_1", "LPD_0", "LPD_1"])}
            result = env.step(action_dict)
            
            # Parse result - handle Gymnasium (5 values) format
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward = result[0], result[1] if len(result) > 1 else 0
                done = False
                info = result[2] if len(result) > 2 else {}
            
            if isinstance(obs, dict):
                obs = np.stack([obs[n] for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
            
            # Get per-agent rewards - prefer info['agent_rewards'] if available
            if isinstance(info, dict) and 'agent_rewards' in info:
                agent_rewards = info['agent_rewards']
                rewards = np.array([agent_rewards.get(n, 0) for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
            elif isinstance(reward, np.ndarray):
                rewards = reward
            elif isinstance(reward, dict):
                rewards = np.array([reward.get(n, 0) for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
            else:
                # Scalar reward - distribute equally
                rewards = np.array([float(reward) / 4] * 4)
            
            collector.record_step(actions, rewards, sim, step)
            
            if isinstance(info, dict) and 'completed_task_latencies' in info:
                for lat in info['completed_task_latencies']:
                    collector.record_task_latency(lat)
            
            if isinstance(done, np.ndarray):
                is_done = done.any()
            elif isinstance(done, dict):
                is_done = any(done.values())
            else:
                is_done = bool(done)
            
            if is_done:
                break
        
        collector.finish_episode(sim)
    
    return collector


# =============================================================================
# MULTI-SCENARIO EVALUATION
# =============================================================================

ALL_SCENARIOS = [
    "easy", "medium", "hard", "streaming_hard", "stress",
    "rural_field", "variable_terrain", "weather_stress", "peak_survey", "network_aware"
]

# Scenarios for generalization testing (more diverse network conditions)
GENERALIZATION_SCENARIOS = [
    "hard",           # Baseline difficult
    "rural_field",    # Low bandwidth
    "variable_terrain",  # High variance
    "network_aware",  # Fast cycling
    "peak_survey",    # Heavy load
]


def run_multi_scenario_evaluation(
    policies: List,
    scenarios: List[str] = None,
    n_episodes: int = 20,
    seed: int = 42,
    comprehensive: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, ComprehensiveMetricCollector]]:
    """
    Evaluate multiple policies across multiple scenarios.
    
    Returns:
        Dict[scenario][policy_name] = ComprehensiveMetricCollector
    """
    from fieldvision_complete import get_difficulty_config, make_multi_drone_env
    
    if scenarios is None:
        scenarios = GENERALIZATION_SCENARIOS
    
    results = {scenario: {} for scenario in scenarios}
    
    for scenario in scenarios:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario}")
            print('='*60)
        
        config = get_difficulty_config(scenario)
        
        def make_env(s):
            return make_multi_drone_env(seed=s, **config)
        
        for policy in policies:
            if verbose:
                print(f"  Evaluating {policy.name}...", end=" ", flush=True)
            
            if comprehensive:
                collector = evaluate_policy_comprehensive(
                    policy, make_env, scenario, n_episodes, max_steps=config.get("max_steps", 100), seed=seed
                )
            else:
                collector = evaluate_policy(policy, make_env, n_episodes, config.get("max_steps", 100), seed)
            
            results[scenario][policy.name] = collector
            
            if verbose:
                s = collector.summary()
                print(f"Reward={s['reward_mean']:.1f}±{s['reward_std']:.1f}")
    
    return results


def run_generalization_test(
    train_scenario: str,
    test_scenarios: List[str] = None,
    total_timesteps: int = 200_000,
    n_eval_episodes: int = 20,
    algorithm: str = "mappo",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Train on one scenario, test on multiple to measure generalization.
    
    Returns:
        Dict with 'train_scenario', 'training_stats', and 'generalization_results'
    """
    from fieldvision_complete import (
        get_difficulty_config, make_multi_drone_env, 
        TrainingConfig, Trainer, MAPPOWrapper
    )
    import torch
    
    if test_scenarios is None:
        test_scenarios = GENERALIZATION_SCENARIOS
    
    # Training
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {algorithm.upper()} on {train_scenario}")
        print('='*60)
    
    train_config = get_difficulty_config(train_scenario)
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        max_steps_per_episode=train_config.get("max_steps", 100)
    )
    
    def make_train_env(seed=None):
        if seed is None:
            seed = np.random.randint(0, 100000)
        return make_multi_drone_env(seed=seed, **train_config)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainer = Trainer(algorithm, config, make_train_env, f"runs/generalization/{algorithm}/{train_scenario}")
    trainer.train()
    
    # Evaluate on training scenario
    train_stats = trainer.evaluate(n_episodes=n_eval_episodes, base_seed=5000)
    
    if verbose:
        print(f"\nTraining scenario performance: {train_stats['reward_mean']:.1f}±{train_stats['reward_std']:.1f}")
    
    # Test on each scenario
    generalization_results = {}
    
    for test_scenario in test_scenarios:
        if verbose:
            print(f"\nTesting on {test_scenario}...", end=" ", flush=True)
        
        test_config = get_difficulty_config(test_scenario)
        
        def make_test_env(seed=None):
            if seed is None:
                seed = np.random.randint(0, 100000)
            return make_multi_drone_env(seed=seed, **test_config)
        
        # Create test environments
        test_envs = [MAPPOWrapper(make_test_env(seed=6000 + i * 100)) for i in range(n_eval_episodes)]
        
        rewards = []
        completions = []
        deadline_misses = []
        transfer_failures = []
        action_counts = {0: 0, 1: 0, 2: 0}
        
        for env in test_envs:
            lo, go, _ = env.reset()
            ep_reward = 0.0
            
            for step in range(test_config.get("max_steps", 100)):
                env.begin_step()
                
                # Get actions from trained policy
                with torch.no_grad():
                    lo_t = torch.tensor(lo, dtype=torch.float32, device=trainer.device)
                    go_t = torch.tensor(go, dtype=torch.float32, device=trainer.device)
                    masks = env.get_action_masks()
                    masks_t = torch.tensor(masks, dtype=torch.float32, device=trainer.device)
                    
                    actions = []
                    for i in range(trainer.n_agents):
                        actor = trainer.get_actor_for_agent(i)
                        if algorithm == "centralized_ppo":
                            obs = go_t.unsqueeze(0)
                        else:
                            obs = lo_t[i].unsqueeze(0)
                        dist = actor(obs, masks_t[i].unsqueeze(0))
                        actions.append(dist.sample().item())
                    actions = np.array(actions)
                
                for a in actions:
                    action_counts[a] += 1
                
                lo, go, r, d, _, info = env.complete_step(actions)
                ep_reward += r.sum()
                
                if d[0]:
                    break
            
            rewards.append(ep_reward)
            sim = env.env.sim
            comp = getattr(sim, 'processed_chunks_count', 0) / max(getattr(sim, 'total_chunks', 1), 1)
            completions.append(comp)
            deadline_misses.append(getattr(sim, 'deadline_misses', 0))
            transfer_failures.append(getattr(sim, 'transfer_failures', 0))
        
        total_actions = sum(action_counts.values())
        generalization_results[test_scenario] = {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'rewards': rewards,
            'completion': np.mean(completions) * 100,
            'completions': [c * 100 for c in completions],  # Per-episode
            'deadline_misses': np.mean(deadline_misses),
            'deadline_misses_list': deadline_misses,  # Per-episode
            'transfer_failures': np.mean(transfer_failures),
            'transfer_failures_list': transfer_failures,  # Per-episode
            'local_pct': action_counts[0] / total_actions * 100 if total_actions > 0 else 0,
            'edge_pct': action_counts[1] / total_actions * 100 if total_actions > 0 else 0,
            'cloud_pct': action_counts[2] / total_actions * 100 if total_actions > 0 else 0,
            'is_train_scenario': test_scenario == train_scenario,
        }
        
        if verbose:
            r = generalization_results[test_scenario]
            marker = " (TRAIN)" if r['is_train_scenario'] else ""
            print(f"{r['reward_mean']:.1f}±{r['reward_std']:.1f}{marker}")
    
    return {
        'algorithm': algorithm,
        'train_scenario': train_scenario,
        'training_stats': train_stats,
        'generalization_results': generalization_results,
    }


def save_comprehensive_results(
    output_dir: str,
    multi_scenario_results: Dict[str, Dict[str, ComprehensiveMetricCollector]] = None,
    generalization_results: Dict = None,
    config: Dict = None,
):
    """
    Save comprehensive evaluation results with per-step data.
    
    Directory structure:
    output_dir/
    ├── config.json
    ├── summary_by_scenario.csv        # All policies × scenarios
    ├── scenarios/
    │   └── {scenario}/
    │       └── {policy}/
    │           ├── episode_metrics.csv
    │           └── step_metrics.csv    # Detailed per-step data!
    └── generalization/
        └── {algorithm}/
            ├── train_info.json
            └── test_results.csv
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    if config:
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
    
    # Save multi-scenario results
    if multi_scenario_results:
        all_summaries = []
        
        for scenario, policy_results in multi_scenario_results.items():
            scenario_dir = path / "scenarios" / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            for policy_name, collector in policy_results.items():
                policy_dir = scenario_dir / policy_name.replace("/", "_").replace(" ", "_")
                policy_dir.mkdir(parents=True, exist_ok=True)
                
                # Episode-level metrics
                collector.to_df().to_csv(policy_dir / "episode_metrics.csv", index=False)
                
                # Step-level metrics (the detailed stuff!)
                if hasattr(collector, 'steps_to_df'):
                    step_df = collector.steps_to_df()
                    if len(step_df) > 0:
                        step_df.to_csv(policy_dir / "step_metrics.csv", index=False)
                
                # Summary
                summary = collector.summary()
                all_summaries.append(summary)
        
        # Combined summary
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(path / "summary_by_scenario.csv", index=False)
        print(f"✓ Saved multi-scenario results to {path}")
    
    # Save generalization results
    if generalization_results:
        gen_dir = path / "generalization" / generalization_results['algorithm']
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        # Training info
        train_info = {
            'algorithm': generalization_results['algorithm'],
            'train_scenario': generalization_results['train_scenario'],
            'training_stats': generalization_results['training_stats'],
        }
        with open(gen_dir / "train_info.json", 'w') as f:
            json.dump(train_info, f, indent=2, cls=NumpyEncoder)
        
        # Test results
        test_rows = []
        for scenario, stats in generalization_results['generalization_results'].items():
            row = {'test_scenario': scenario, **stats}
            del row['rewards']  # Don't include raw list in CSV
            test_rows.append(row)
        
        test_df = pd.DataFrame(test_rows)
        test_df.to_csv(gen_dir / "test_results.csv", index=False)
        
        # Also save raw rewards for statistical tests
        rewards_data = {scenario: stats['rewards'] 
                       for scenario, stats in generalization_results['generalization_results'].items()}
        with open(gen_dir / "test_rewards.json", 'w') as f:
            json.dump(rewards_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved generalization results to {gen_dir}")


def print_multi_scenario_summary(results: Dict[str, Dict[str, ComprehensiveMetricCollector]]):
    """Print a nice summary table of multi-scenario results."""
    print("\n" + "=" * 80)
    print("MULTI-SCENARIO EVALUATION SUMMARY")
    print("=" * 80)
    
    # Get all policies
    all_policies = set()
    for scenario_results in results.values():
        all_policies.update(scenario_results.keys())
    policies = sorted(all_policies)
    
    # Header
    print(f"{'Policy':<25}", end="")
    for scenario in results.keys():
        print(f"{scenario[:12]:>14}", end="")
    print()
    print("-" * 80)
    
    # Rows
    for policy in policies:
        print(f"{policy:<25}", end="")
        for scenario, scenario_results in results.items():
            if policy in scenario_results:
                s = scenario_results[policy].summary()
                print(f"{s['reward_mean']:>8.1f}±{s['reward_std']:<4.0f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()
    
    # Find best per scenario
    print("-" * 80)
    print(f"{'BEST':<25}", end="")
    for scenario, scenario_results in results.items():
        best_policy = max(scenario_results.keys(), 
                         key=lambda p: scenario_results[p].summary()['reward_mean'])
        best_reward = scenario_results[best_policy].summary()['reward_mean']
        print(f"{best_reward:>14.1f}", end="")
    print()


def print_generalization_summary(gen_results: Dict):
    """Print generalization test summary."""
    print("\n" + "=" * 60)
    print(f"GENERALIZATION TEST: {gen_results['algorithm'].upper()}")
    print(f"Trained on: {gen_results['train_scenario']}")
    print("=" * 60)
    
    train_r = gen_results['training_stats']['reward_mean']
    
    print(f"\n{'Scenario':<20} {'Reward':>12} {'vs Train':>12} {'Completion':>12}")
    print("-" * 60)
    
    for scenario, stats in gen_results['generalization_results'].items():
        marker = " *" if stats['is_train_scenario'] else ""
        diff = stats['reward_mean'] - train_r
        diff_pct = diff / abs(train_r) * 100 if train_r != 0 else 0
        print(f"{scenario:<20} {stats['reward_mean']:>8.1f}±{stats['reward_std']:<3.0f} "
              f"{diff:>+8.1f} ({diff_pct:>+5.1f}%) {stats['completion']:>8.1f}%{marker}")
    
    print("\n* = training scenario")


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def evaluate_policy(policy, make_env_fn, n_episodes=30, max_steps=300, seed=42) -> MetricCollector:
    """Evaluate a single policy."""
    collector = MetricCollector(policy.name)
    
    for ep in range(n_episodes):
        env = make_env_fn(seed + ep)
        
        # Handle both wrapper and raw env
        if hasattr(env, 'reset') and callable(env.reset):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
        
        # Convert obs if dict
        if isinstance(obs, dict):
            obs = np.stack([obs[n] for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
        
        policy.reset()
        collector.reset_episode()
        ep_reward = 0.0
        
        # Get sim reference
        sim = env.sim if hasattr(env, 'sim') else (env.env.sim if hasattr(env, 'env') else env)
        
        for step in range(max_steps):
            actions = policy.get_action(obs, sim)
            
            # Convert actions to dict format for env.step
            action_dict = {n: int(actions[i]) for i, n in enumerate(["MPD_0", "MPD_1", "LPD_0", "LPD_1"])}
            result = env.step(action_dict)
            
            # Parse result - handle both Gymnasium (5 values) and older (4 values) formats
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                # Fallback for unexpected formats
                obs, reward = result[0], result[1] if len(result) > 1 else 0
                done = False
                info = result[2] if len(result) > 2 else {}
            
            # Record task latencies from info
            if isinstance(info, dict) and 'completed_task_latencies' in info:
                for lat in info['completed_task_latencies']:
                    collector.record_task_latency(lat)
            
            # Convert obs
            if isinstance(obs, dict):
                obs = np.stack([obs[n] for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]])
            
            # Handle reward - DON'T multiply by n_agents, sum is already total
            if isinstance(reward, np.ndarray):
                r = float(reward.sum())
            elif isinstance(reward, dict):
                r = sum(reward.values())
            else:
                r = float(reward)
            
            ep_reward += r
            
            # Get bandwidth for metrics
            bw = 5.0
            if hasattr(sim, 'drone_network_states') and sim.drone_network_states:
                bws = [sim.drone_network_states[n].drone_ground_bandwidth_mbps 
                       for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"] if n in sim.drone_network_states]
                if bws:
                    bw = np.mean(bws)
            
            collector.step(actions, bw)
            
            # Check done
            if isinstance(done, np.ndarray):
                is_done = done.any()
            elif isinstance(done, dict):
                is_done = any(done.values())
            else:
                is_done = bool(done)
            
            if is_done:
                break
        
        collector.finish_episode(ep, ep_reward, step + 1, sim)
    
    return collector

def evaluate_all_baselines(difficulty: str, n_episodes: int = 30, seed: int = 42, 
                          make_env_fn=None, verbose=True) -> Dict[str, MetricCollector]:
    """Evaluate all baseline policies."""
    # Import here to avoid circular imports
    try:
        from fieldvision_complete import make_multi_drone_env, get_difficulty_config
    except ImportError:
        raise ImportError("fieldvision_complete.py required")
    
    config = get_difficulty_config(difficulty)
    
    if make_env_fn is None:
        def make_env_fn(s):
            # Don't wrap in MAPPOWrapper - baselines use raw environment
            return make_multi_drone_env(seed=s, **config)
    
    if verbose:
        print("=" * 60)
        print(f"BASELINE EVALUATION - {difficulty}")
        print(f"Episodes: {n_episodes}, Seed: {seed}")
        print("=" * 60)
    
    results = {}
    policies = get_all_policies(seed)
    
    for policy in policies:
        if verbose:
            print(f"\nEvaluating {policy.name}...")
        
        collector = evaluate_policy(policy, make_env_fn, n_episodes, config.get("max_steps", 300), seed)
        results[policy.name] = collector
        
        if verbose:
            s = collector.summary()
            print(f"  Reward: {s['reward_mean']:.1f} ± {s['reward_std']:.1f}")
            print(f"  Actions: L={s['local_pct']:.0f}% E={s['edge_pct']:.0f}% C={s['cloud_pct']:.0f}%")
    
    return results


def evaluate_marl_agents(trainers: Dict, make_env_fn, n_episodes: int = 30, 
                         max_steps: int = 300, base_seed: int = 2000,
                         verbose: bool = True) -> Dict[str, MetricCollector]:
    """
    Evaluate trained MARL agents and return MetricCollector objects for figures.
    
    Args:
        trainers: Dict mapping algorithm names to trained Trainer objects
        make_env_fn: Function that creates environment given a seed
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        base_seed: Starting seed for evaluation
        verbose: Whether to print progress
        
    Returns:
        Dict mapping algorithm names to MetricCollector objects
    """
    try:
        import torch
        from fieldvision_complete import MAPPOWrapper
    except ImportError:
        raise ImportError("torch and fieldvision_complete required")
    
    results = {}
    
    for alg_name, trainer in trainers.items():
        if verbose:
            print(f"\nEvaluating {alg_name}...")
        
        collector = MetricCollector(alg_name)
        
        for ep in range(n_episodes):
            env = MAPPOWrapper(make_env_fn(seed=base_seed + ep))
            lo, go, _ = env.reset()
            sim = env.sim
            
            lo_t = torch.tensor(lo, dtype=torch.float32, device=trainer.device)
            go_t = torch.tensor(go, dtype=torch.float32, device=trainer.device)
            
            collector.reset_episode()
            ep_reward = 0.0
            done = False
            step = 0
            
            while not done and step < max_steps:
                with torch.no_grad():
                    actions = []
                    masks = torch.tensor(env.get_action_masks(), dtype=torch.bool, device=trainer.device)
                    
                    for i in range(trainer.n_agents):
                        actor = trainer.get_actor_for_agent(i)
                        obs = go_t.unsqueeze(0) if trainer.algorithm == "centralized_ppo" else lo_t[i].unsqueeze(0)
                        action = actor(obs, masks[i].unsqueeze(0)).sample().item()
                        actions.append(action)
                
                actions_np = np.array(actions)
                lo, go, r, d, info = env.step(actions_np)
                
                step_reward = r.sum()
                ep_reward += step_reward
                
                # Record task latencies from completed tasks
                if 'completed_task_latencies' in info:
                    for lat in info['completed_task_latencies']:
                        collector.record_task_latency(lat)
                
                # Get bandwidth for metrics
                bw = 5.0
                if hasattr(sim, 'drone_network_states') and sim.drone_network_states:
                    bws = [sim.drone_network_states[n].drone_ground_bandwidth_mbps 
                           for n in ["MPD_0", "MPD_1", "LPD_0", "LPD_1"] if n in sim.drone_network_states]
                    if bws:
                        bw = np.mean(bws)
                
                collector.step(actions_np, bw)
                
                done = d[0]
                lo_t = torch.tensor(lo, dtype=torch.float32, device=trainer.device)
                go_t = torch.tensor(go, dtype=torch.float32, device=trainer.device)
                step += 1
            
            collector.finish_episode(ep, ep_reward, step, sim)
        
        results[alg_name] = collector
        
        if verbose:
            s = collector.summary()
            print(f"  Reward: {s['reward_mean']:.1f} ± {s['reward_std']:.1f}")
            print(f"  Actions: L={s['local_pct']:.0f}% E={s['edge_pct']:.0f}% C={s['cloud_pct']:.0f}%")
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_all_figures(results: Dict[str, MetricCollector], output_dir: str = "figures/",
                        marl_results: Optional[Dict] = None,
                        exclude_outliers: bool = True,
                        outlier_policies: Optional[List[str]] = None):
    """Generate publication-quality figures including MARL results.
    
    Args:
        results: Baseline policy results (MetricCollector dict)
        output_dir: Directory to save figures
        marl_results: MARL results (dict of dicts from run_comparison_experiment)
        exclude_outliers: If True, exclude extreme outliers from graphs
        outlier_policies: List of policy names to exclude (default: ["Local-Only"])
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib required for figures")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Default outlier policies
    if outlier_policies is None:
        outlier_policies = ["Local-Only"]
    
    # Convert MARL results to adapter objects
    marl_adapted = {}
    if marl_results:
        marl_adapted = convert_marl_results(marl_results)
    
    # Combine baseline and MARL results
    all_results = dict(results)
    all_results.update(marl_adapted)
    
    # Filter out outliers for graphs (but keep them in all_results for complete data)
    if exclude_outliers:
        filtered_results = {k: v for k, v in all_results.items() if k not in outlier_policies}
    else:
        filtered_results = all_results
    
    # Prepare data
    summaries = [c.summary() for c in filtered_results.values()]
    df = pd.DataFrame(summaries).sort_values("reward_mean", ascending=False)
    
    # Also create full summary (including outliers)
    all_summaries = [c.summary() for c in all_results.values()]
    df_all = pd.DataFrame(all_summaries).sort_values("reward_mean", ascending=False)
    
    # Color scheme: baselines in blue spectrum, MARL in red/orange spectrum
    baseline_names = [k for k in results.keys() if k not in outlier_policies]
    marl_names = list(marl_adapted.keys())
    
    n_baseline = len(baseline_names)
    n_marl = len(marl_names)
    
    baseline_colors = plt.cm.Blues(np.linspace(0.4, 0.8, max(n_baseline, 1)))
    marl_colors = plt.cm.Oranges(np.linspace(0.5, 0.9, max(n_marl, 1)))
    
    # Map colors to policies
    color_map = {}
    for i, name in enumerate(baseline_names):
        color_map[name] = baseline_colors[i % len(baseline_colors)]
    for i, name in enumerate(marl_names):
        color_map[name] = marl_colors[i % len(marl_colors)]
    # Gray for outliers
    for name in outlier_policies:
        color_map[name] = 'lightgray'
    
    # Figure 1: Reward Comparison (bar chart with error bars)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [color_map.get(p, 'gray') for p in df["policy"]]
    bars = ax.bar(range(len(df)), df["reward_mean"], yerr=df["reward_std"], 
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["policy"], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Policy Comparison - Episode Rewards", fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, df["reward_mean"]):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Add legend for baseline vs MARL
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.Blues(0.6), label='Baseline'),
                       Patch(facecolor=plt.cm.Oranges(0.7), label='MARL')]
    if marl_adapted:
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Action Distribution (pie charts)
    n_policies = len(filtered_results)
    n_cols = min(4, n_policies)
    n_rows = (n_policies + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_policies == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (name, collector) in enumerate(filtered_results.items()):
        if idx >= len(axes):
            break
        s = collector.summary()
        if s["local_pct"] is None or (s["local_pct"] == 0 and s["edge_pct"] == 0 and s["cloud_pct"] == 0):
            # Skip if no action data
            axes[idx].text(0.5, 0.5, f"{name}\n(no action data)", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
            continue
            
        sizes = [s["local_pct"], s["edge_pct"], s["cloud_pct"]]
        labels = ["Local", "Edge", "Cloud"]
        colors_pie = ["#2ecc71", "#3498db", "#e74c3c"]
        
        # Bold title for MARL policies
        is_marl = name in marl_adapted
        wedges, texts, autotexts = axes[idx].pie(sizes, labels=labels, colors=colors_pie, 
                                                  autopct='%1.0f%%', startangle=90, 
                                                  textprops={'fontsize': 9})
        title_style = {'fontsize': 10, 'fontweight': 'bold', 'color': 'darkorange'} if is_marl else {'fontsize': 10}
        axes[idx].set_title(name, **title_style)
    
    for idx in range(len(filtered_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Action Distribution by Policy", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Completion Rate
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter out policies with no completion data
    df_completion = df[df["completion"].notna() & (df["completion"] > 0)]
    if len(df_completion) > 0:
        colors = [color_map.get(p, 'gray') for p in df_completion["policy"]]
        ax.bar(range(len(df_completion)), df_completion["completion"], color=colors, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(df_completion)))
        ax.set_xticklabels(df_completion["policy"], rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Completion Rate (%)", fontsize=12)
        ax.set_title("Task Completion Rate by Policy", fontsize=14)
        ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/completion_rate.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Task Latency CDF (time from task creation to completion in timesteps)
    fig, ax = plt.subplots(figsize=(10, 6))
    has_latency_data = False
    
    for name, collector in filtered_results.items():
        if hasattr(collector, 'all_task_latencies') and collector.all_task_latencies and len(collector.all_task_latencies) > 0:
            has_latency_data = True
            latencies = np.array(collector.all_task_latencies)
            sorted_latencies = np.sort(latencies)
            cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
            
            color = color_map.get(name, 'gray')
            is_marl = name in marl_adapted
            linestyle = '--' if is_marl else '-'
            linewidth = 2.5 if is_marl else 1.5
            ax.plot(sorted_latencies, cdf, label=name, color=color, 
                   linestyle=linestyle, linewidth=linewidth)
    
    if has_latency_data:
        ax.set_xlabel("Task Latency (timesteps from creation to completion)", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.set_title("Cumulative Distribution of Task Latencies", fontsize=14)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_cdf.png", dpi=150, bbox_inches='tight')
    else:
        ax.text(0.5, 0.5, "No task latency data available", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        plt.savefig(f"{output_dir}/latency_cdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Reward Distribution (box plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    reward_data = []
    labels = []
    colors_box = []
    for name, collector in filtered_results.items():
        if hasattr(collector, 'episodes') and collector.episodes:
            rewards = [e.total_reward for e in collector.episodes]
            if rewards:
                reward_data.append(rewards)
                labels.append(name)
                colors_box.append(color_map.get(name, 'gray'))
    
    if reward_data:
        bp = ax.boxplot(reward_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title("Reward Distribution by Policy", fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {output_dir}/")
    print(f"  - reward_comparison.png")
    print(f"  - action_distribution.png") 
    print(f"  - completion_rate.png")
    print(f"  - latency_cdf.png")
    print(f"  - reward_boxplot.png")
    if exclude_outliers and outlier_policies:
        print(f"  (Excluded from graphs: {outlier_policies})")
    
    return df_all  # Return full summary including outliers

# =============================================================================
# CHECKPOINT & REPLICATION
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def save_checkpoint(output_dir: str, results: Dict[str, MetricCollector], 
                   config: Optional[Dict] = None, marl_results: Optional[Dict] = None):
    """Save comprehensive experiment checkpoint for replication and further analysis.
    
    Saved data structure:
    output_dir/
    ├── config.json                     # Experiment configuration
    ├── summary.csv                     # All policies sorted by reward
    ├── baseline_results/
    │   └── {policy_name}/
    │       └── episode_metrics.csv     # Per-episode metrics
    ├── marl_results/
    │   └── {algorithm}/
    │       ├── summary.json            # Aggregate stats
    │       └── episode_rewards.csv     # Per-episode rewards
    └── analysis/
        ├── reward_comparison.csv       # Side-by-side reward comparison
        └── action_distribution.csv     # Action % by policy
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    if config:
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
        print(f"✓ Saved config.json")
    
    # Save baseline results
    baseline_path = path / "baseline_results"
    for name, collector in results.items():
        policy_path = baseline_path / name.replace(" ", "_").replace("-", "_").replace("/", "_")
        policy_path.mkdir(parents=True, exist_ok=True)
        collector.to_df().to_csv(policy_path / "episode_metrics.csv", index=False)
        
        # Save task latencies if available
        if hasattr(collector, 'all_task_latencies') and collector.all_task_latencies:
            latency_df = pd.DataFrame({
                "latency_timesteps": collector.all_task_latencies
            })
            latency_df.to_csv(policy_path / "task_latencies.csv", index=False)
    print(f"✓ Saved baseline_results/")
    
    # Save MARL results
    if marl_results:
        marl_path = path / "marl_results"
        marl_path.mkdir(parents=True, exist_ok=True)
        
        for alg, data in marl_results.items():
            alg_path = marl_path / alg
            alg_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, (MetricCollector, MARLResultsAdapter)):
                data.to_df().to_csv(alg_path / "episode_metrics.csv", index=False)
                # Also save summary
                summary = data.summary()
                with open(alg_path / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, cls=NumpyEncoder)
            elif isinstance(data, dict):
                # New format: dict with reward_mean, reward_std, rewards, etc.
                # Create adapter to get full episode metrics
                adapter = MARLResultsAdapter(alg, data)
                adapter.to_df().to_csv(alg_path / "episode_metrics.csv", index=False)
                
                summary = {
                    "algorithm": alg,
                    "reward_mean": float(data.get('reward_mean', 0)),
                    "reward_std": float(data.get('reward_std', 0)),
                    "completion": float(data.get('completion', 0)),
                    "deadline_misses": float(data.get('deadline_misses', 0)),
                    "transfer_failures": float(data.get('transfer_failures', 0)),
                    "local_pct": float(data.get('local_pct', 0)),
                    "edge_pct": float(data.get('edge_pct', 0)),
                    "cloud_pct": float(data.get('cloud_pct', 0)),
                    "n_episodes": len(data.get('rewards', [])),
                }
                with open(alg_path / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, cls=NumpyEncoder)
            elif isinstance(data, list):
                # Old format: list of rewards
                native_rewards = [float(r) for r in data]
                rewards_df = pd.DataFrame({
                    "seed": list(range(len(native_rewards))),
                    "eval_reward": native_rewards
                })
                rewards_df.to_csv(alg_path / "eval_rewards.csv", index=False)
                
                summary = {
                    "algorithm": alg,
                    "reward_mean": float(np.mean(native_rewards)),
                    "reward_std": float(np.std(native_rewards)),
                    "n_seeds": len(native_rewards),
                }
                with open(alg_path / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved marl_results/")
    
    # Save combined summary
    summaries = [c.summary() for c in results.values()]
    
    # Add MARL results to summary
    if marl_results:
        marl_adapted = convert_marl_results(marl_results)
        for alg, adapter in marl_adapted.items():
            summaries.append(adapter.summary())
    
    summary_df = pd.DataFrame(summaries).sort_values("reward_mean", ascending=False)
    summary_df.to_csv(path / "summary.csv", index=False)
    print(f"✓ Saved summary.csv")
    
    # Save analysis-ready files
    analysis_path = path / "analysis"
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    # Reward comparison (wide format for easy plotting)
    reward_data = []
    for name, collector in results.items():
        for ep in collector.episodes:
            reward_data.append({
                "policy": name,
                "type": "baseline",
                "episode": ep.episode,
                "reward": ep.total_reward,
                "completion_rate": ep.completion_rate,
                "deadline_misses": ep.deadline_misses,
                "transfer_failures": ep.transfer_failures,
            })
    
    if marl_results:
        marl_adapted = convert_marl_results(marl_results)
        for alg, adapter in marl_adapted.items():
            for ep in adapter.episodes:
                reward_data.append({
                    "policy": alg,
                    "type": "marl",
                    "episode": ep.episode,
                    "reward": ep.total_reward,
                    "completion_rate": ep.completion_rate,
                    "deadline_misses": ep.deadline_misses,
                    "transfer_failures": ep.transfer_failures,
                })
    
    reward_df = pd.DataFrame(reward_data)
    reward_df.to_csv(analysis_path / "all_episode_rewards.csv", index=False)
    
    # Action distribution summary
    action_data = []
    for s in summaries:
        action_data.append({
            "policy": s["policy"],
            "local_pct": s.get("local_pct", 0),
            "edge_pct": s.get("edge_pct", 0),
            "cloud_pct": s.get("cloud_pct", 0),
        })
    action_df = pd.DataFrame(action_data)
    action_df.to_csv(analysis_path / "action_distribution.csv", index=False)
    
    print(f"✓ Saved analysis/ (for further analysis)")
    
    print(f"\nCheckpoint saved to: {output_dir}")
    print(f"\nFiles saved:")
    print(f"  - config.json: Experiment configuration")
    print(f"  - summary.csv: All policies ranked by reward")
    print(f"  - baseline_results/: Per-episode metrics for each baseline")
    print(f"  - marl_results/: MARL algorithm results")
    print(f"  - analysis/all_episode_rewards.csv: All episode rewards (long format)")
    print(f"  - analysis/action_distribution.csv: Action % by policy")
    
    return summary_df

def load_checkpoint(input_dir: str):
    """Load experiment checkpoint."""
    path = Path(input_dir)
    
    config = None
    if (path / "config.json").exists():
        with open(path / "config.json") as f:
            config = json.load(f)
    
    summary = pd.read_csv(path / "summary.csv") if (path / "summary.csv").exists() else None
    
    results = {}
    baseline_path = path / "baseline_results"
    if baseline_path.exists():
        for policy_dir in baseline_path.iterdir():
            if policy_dir.is_dir():
                metrics_file = policy_dir / "episode_metrics.csv"
                if metrics_file.exists():
                    results[policy_dir.name] = pd.read_csv(metrics_file)
    
    return config, summary, results

# =============================================================================
# COMPREHENSIVE RESULTS AGGREGATION AND VISUALIZATION
# =============================================================================

def aggregate_marl_results(marl_results: Dict) -> pd.DataFrame:
    """
    Aggregate MARL results into a comprehensive DataFrame for analysis.
    
    Args:
        marl_results: Dict from run_comparison_experiment or similar
                     {alg_name: {metric: value, ...}, ...}
    
    Returns:
        DataFrame with one row per algorithm, all metrics as columns
    """
    rows = []
    for alg, stats in marl_results.items():
        row = {
            'algorithm': alg,
            'reward_mean': stats.get('reward_mean', 0),
            'reward_std': stats.get('reward_std', 0),
            'completion': stats.get('completion', 0),
            'on_time_rate': stats.get('on_time_rate', 100),
            'deadline_misses': stats.get('deadline_misses', 0),
            'transfer_failures': stats.get('transfer_failures', 0),
            'avg_latency': stats.get('avg_latency', 0),
            'latency_std': stats.get('latency_std', 0),
            'final_battery_mean': stats.get('final_battery_mean', 1.0),
            'final_battery_mpd': stats.get('final_battery_mpd', 1.0),
            'final_battery_lpd': stats.get('final_battery_lpd', 1.0),
            'battery_used_pct': stats.get('battery_used_pct', 0),
            'local_pct': stats.get('local_pct', 0),
            'edge_pct': stats.get('edge_pct', 0),
            'cloud_pct': stats.get('cloud_pct', 0),
            'n_episodes': len(stats.get('rewards', [])),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_summary_table(marl_results: Dict, baseline_results: Dict = None,
                         output_path: str = None) -> pd.DataFrame:
    """
    Create a comprehensive summary table suitable for publication.
    
    Args:
        marl_results: MARL algorithm results
        baseline_results: Optional baseline policy results (MetricCollector objects)
        output_path: Optional path to save CSV
    
    Returns:
        DataFrame with all algorithms and key metrics
    """
    rows = []
    
    # Add baseline results
    if baseline_results:
        for name, collector in baseline_results.items():
            if hasattr(collector, 'summary'):
                s = collector.summary()
            elif hasattr(collector, 'to_df'):
                df = collector.to_df()
                s = {
                    'reward_mean': df['total_reward'].mean(),
                    'reward_std': df['total_reward'].std(),
                    'completion': df['completion_rate'].mean() * 100,
                    'on_time_rate': df['on_time_rate'].mean() if 'on_time_rate' in df else 100,
                    'deadline_misses': df['deadline_misses'].mean() if 'deadline_misses' in df else 0,
                    'local_pct': df['local_pct'].mean() if 'local_pct' in df else 0,
                    'edge_pct': df['edge_pct'].mean() if 'edge_pct' in df else 0,
                    'cloud_pct': df['cloud_pct'].mean() if 'cloud_pct' in df else 0,
                }
            else:
                continue
            
            rows.append({
                'Algorithm': name,
                'Type': 'Baseline',
                'Reward': f"{s.get('reward_mean', 0):.1f} ± {s.get('reward_std', 0):.1f}",
                'Completion %': f"{s.get('completion', 0):.1f}",
                'On-Time %': f"{s.get('on_time_rate', 100):.1f}",
                'Deadline Misses': f"{s.get('deadline_misses', 0):.2f}",
                'Avg Latency': f"{s.get('avg_latency', 0):.1f}",
                'Battery Used %': f"{s.get('battery_used_pct', 0):.1f}",
                'Local %': f"{s.get('local_pct', 0):.0f}",
                'Edge %': f"{s.get('edge_pct', 0):.0f}",
                'Cloud %': f"{s.get('cloud_pct', 0):.0f}",
            })
    
    # Add MARL results
    for alg, stats in marl_results.items():
        rows.append({
            'Algorithm': alg,
            'Type': 'MARL',
            'Reward': f"{stats.get('reward_mean', 0):.1f} ± {stats.get('reward_std', 0):.1f}",
            'Completion %': f"{stats.get('completion', 0):.1f}",
            'On-Time %': f"{stats.get('on_time_rate', 100):.1f}",
            'Deadline Misses': f"{stats.get('deadline_misses', 0):.2f}",
            'Avg Latency': f"{stats.get('avg_latency', 0):.1f}",
            'Battery Used %': f"{stats.get('battery_used_pct', 0):.1f}",
            'Local %': f"{stats.get('local_pct', 0):.0f}",
            'Edge %': f"{stats.get('edge_pct', 0):.0f}",
            'Cloud %': f"{stats.get('cloud_pct', 0):.0f}",
        })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Saved summary table to {output_path}")
    
    return df


def create_latex_table(marl_results: Dict, baseline_results: Dict = None,
                       output_path: str = None, caption: str = "Algorithm Comparison") -> str:
    """
    Generate a LaTeX table for publication.
    
    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Algorithm & Reward & Completion & On-Time & Latency & Battery & Action Dist. \\",
        r" & (mean±std) & (\%) & (\%) & (steps) & Used (\%) & (L/E/C \%) \\",
        r"\midrule",
    ]
    
    # Baselines
    if baseline_results:
        for name, collector in baseline_results.items():
            if hasattr(collector, 'summary'):
                s = collector.summary()
            elif hasattr(collector, 'to_df'):
                df = collector.to_df()
                s = {
                    'reward_mean': df['total_reward'].mean(),
                    'reward_std': df['total_reward'].std(),
                    'completion': df['completion_rate'].mean() * 100,
                    'on_time_rate': df['on_time_rate'].mean() if 'on_time_rate' in df else 100,
                    'avg_latency': 0,
                    'battery_used_pct': 0,
                    'local_pct': df['local_pct'].mean() if 'local_pct' in df else 0,
                    'edge_pct': df['edge_pct'].mean() if 'edge_pct' in df else 0,
                    'cloud_pct': df['cloud_pct'].mean() if 'cloud_pct' in df else 0,
                }
            else:
                continue
            
            lines.append(
                f"{name} & {s.get('reward_mean',0):.1f}±{s.get('reward_std',0):.1f} & "
                f"{s.get('completion',0):.1f} & {s.get('on_time_rate',100):.1f} & "
                f"{s.get('avg_latency',0):.1f} & {s.get('battery_used_pct',0):.1f} & "
                f"{s.get('local_pct',0):.0f}/{s.get('edge_pct',0):.0f}/{s.get('cloud_pct',0):.0f} \\\\"
            )
    
    lines.append(r"\midrule")
    
    # MARL algorithms
    for alg, stats in marl_results.items():
        alg_display = alg.replace('_', ' ').title()
        lines.append(
            f"{alg_display} & {stats.get('reward_mean',0):.1f}±{stats.get('reward_std',0):.1f} & "
            f"{stats.get('completion',0):.1f} & {stats.get('on_time_rate',100):.1f} & "
            f"{stats.get('avg_latency',0):.1f} & {stats.get('battery_used_pct',0):.1f} & "
            f"{stats.get('local_pct',0):.0f}/{stats.get('edge_pct',0):.0f}/{stats.get('cloud_pct',0):.0f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"✓ Saved LaTeX table to {output_path}")
    
    return latex


def generate_comprehensive_figures(marl_results: Dict, baseline_results: Dict = None,
                                    output_dir: str = "figures"):
    """
    Generate publication-quality figures for all key metrics.
    
    Creates:
    1. Bar chart: Reward comparison
    2. Bar chart: Completion rate comparison
    3. Bar chart: On-time rate (deadline compliance)
    4. Bar chart: Average latency
    5. Bar chart: Battery usage
    6. Stacked bar: Action distribution
    7. Box plots: Reward distribution (if per-episode data available)
    8. Multi-metric radar chart
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available - skipping figure generation")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all algorithms
    all_algs = {}
    alg_types = {}
    
    if baseline_results:
        for name, collector in baseline_results.items():
            if hasattr(collector, 'summary'):
                s = collector.summary()
            elif hasattr(collector, 'to_df'):
                df = collector.to_df()
                s = {
                    'reward_mean': df['total_reward'].mean(),
                    'reward_std': df['total_reward'].std(),
                    'completion': df['completion_rate'].mean() * 100,
                    'on_time_rate': df['on_time_rate'].mean() if 'on_time_rate' in df else 100,
                    'deadline_misses': df['deadline_misses'].mean() if 'deadline_misses' in df else 0,
                    'avg_latency': 0,
                    'battery_used_pct': 0,
                    'local_pct': df['local_pct'].mean() if 'local_pct' in df else 0,
                    'edge_pct': df['edge_pct'].mean() if 'edge_pct' in df else 0,
                    'cloud_pct': df['cloud_pct'].mean() if 'cloud_pct' in df else 0,
                }
            else:
                continue
            all_algs[name] = s
            alg_types[name] = 'baseline'
    
    for alg, stats in marl_results.items():
        all_algs[alg] = stats
        alg_types[alg] = 'marl'
    
    names = list(all_algs.keys())
    colors = ['#3498db' if alg_types[n] == 'baseline' else '#e74c3c' for n in names]
    
    # 1. Reward Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = [all_algs[n].get('reward_mean', 0) for n in names]
    stds = [all_algs[n].get('reward_std', 0) for n in names]
    bars = ax.bar(names, rewards, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('Total Reward')
    ax.set_title('Algorithm Performance: Total Reward')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison.png", dpi=150)
    plt.close()
    
    # 2. Completion Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    completions = [all_algs[n].get('completion', 0) for n in names]
    ax.bar(names, completions, color=colors, alpha=0.8)
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Task Completion Rate')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/completion_comparison.png", dpi=150)
    plt.close()
    
    # 3. On-Time Rate (Deadline Compliance)
    fig, ax = plt.subplots(figsize=(10, 6))
    on_time = [all_algs[n].get('on_time_rate', 100) for n in names]
    ax.bar(names, on_time, color=colors, alpha=0.8)
    ax.set_ylabel('On-Time Rate (%)')
    ax.set_title('Deadline Compliance: Tasks Completed On-Time')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ontime_comparison.png", dpi=150)
    plt.close()
    
    # 4. Average Latency
    fig, ax = plt.subplots(figsize=(10, 6))
    latencies = [all_algs[n].get('avg_latency', 0) for n in names]
    latency_stds = [all_algs[n].get('latency_std', 0) for n in names]
    ax.bar(names, latencies, yerr=latency_stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('Average Latency (timesteps)')
    ax.set_title('Task Processing Latency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=150)
    plt.close()
    
    # 5. Battery Usage
    fig, ax = plt.subplots(figsize=(10, 6))
    battery = [all_algs[n].get('battery_used_pct', 0) for n in names]
    ax.bar(names, battery, color=colors, alpha=0.8)
    ax.set_ylabel('Battery Used (%)')
    ax.set_title('Drone Battery Consumption')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/battery_comparison.png", dpi=150)
    plt.close()
    
    # 6. Action Distribution (Stacked Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    local = [all_algs[n].get('local_pct', 0) for n in names]
    edge = [all_algs[n].get('edge_pct', 0) for n in names]
    cloud = [all_algs[n].get('cloud_pct', 0) for n in names]
    
    x = np.arange(len(names))
    width = 0.6
    ax.bar(x, local, width, label='Local', color='#2ecc71')
    ax.bar(x, edge, width, bottom=local, label='Edge', color='#f39c12')
    ax.bar(x, cloud, width, bottom=[l+e for l,e in zip(local, edge)], label='Cloud', color='#9b59b6')
    
    ax.set_ylabel('Action Distribution (%)')
    ax.set_title('Offloading Decision Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_distribution.png", dpi=150)
    plt.close()
    
    # 7. Multi-metric comparison (grouped bar)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = [
        ('reward_mean', 'Reward', None),
        ('completion', 'Completion %', (0, 105)),
        ('on_time_rate', 'On-Time %', (0, 105)),
        ('avg_latency', 'Avg Latency', None),
        ('battery_used_pct', 'Battery Used %', (0, 100)),
        ('deadline_misses', 'Deadline Misses', None),
    ]
    
    for ax, (metric, title, ylim) in zip(axes.flat, metrics):
        values = [all_algs[n].get(metric, 0) for n in names]
        ax.bar(names, values, color=colors, alpha=0.8)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(ylim)
        ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    baseline_patch = mpatches.Patch(color='#3498db', label='Baseline')
    marl_patch = mpatches.Patch(color='#e74c3c', label='MARL')
    fig.legend(handles=[baseline_patch, marl_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_metric_comparison.png", dpi=150)
    plt.close()
    
    print(f"✓ Generated figures in {output_dir}/")
    print("  - reward_comparison.png")
    print("  - completion_comparison.png")
    print("  - ontime_comparison.png")
    print("  - latency_comparison.png")
    print("  - battery_comparison.png")
    print("  - action_distribution.png")
    print("  - multi_metric_comparison.png")


def print_results_summary(marl_results: Dict, baseline_results: Dict = None):
    """Print a formatted summary of all results to console."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Algorithm':<20} {'Reward':>15} {'Comp%':>8} {'OnTime%':>8} {'Latency':>10} {'Battery%':>10} {'Actions (L/E/C)'}")
    print("-"*95)
    
    # Baselines
    if baseline_results:
        print("\n[BASELINES]")
        for name, collector in baseline_results.items():
            if hasattr(collector, 'summary'):
                s = collector.summary()
            elif hasattr(collector, 'to_df'):
                df = collector.to_df()
                s = {
                    'reward_mean': df['total_reward'].mean(),
                    'reward_std': df['total_reward'].std(),
                    'completion': df['completion_rate'].mean() * 100,
                    'on_time_rate': df['on_time_rate'].mean() if 'on_time_rate' in df else 100,
                    'avg_latency': 0,
                    'battery_used_pct': 0,
                    'local_pct': df['local_pct'].mean() if 'local_pct' in df else 0,
                    'edge_pct': df['edge_pct'].mean() if 'edge_pct' in df else 0,
                    'cloud_pct': df['cloud_pct'].mean() if 'cloud_pct' in df else 0,
                }
            else:
                continue
            
            reward_str = f"{s.get('reward_mean',0):.1f}±{s.get('reward_std',0):.1f}"
            action_str = f"{s.get('local_pct',0):.0f}/{s.get('edge_pct',0):.0f}/{s.get('cloud_pct',0):.0f}"
            print(f"{name:<20} {reward_str:>15} {s.get('completion',0):>7.1f}% {s.get('on_time_rate',100):>7.1f}% "
                  f"{s.get('avg_latency',0):>9.1f} {s.get('battery_used_pct',0):>9.1f}% {action_str}")
    
    # MARL
    print("\n[MARL ALGORITHMS]")
    for alg, stats in marl_results.items():
        reward_str = f"{stats.get('reward_mean',0):.1f}±{stats.get('reward_std',0):.1f}"
        action_str = f"{stats.get('local_pct',0):.0f}/{stats.get('edge_pct',0):.0f}/{stats.get('cloud_pct',0):.0f}"
        print(f"{alg:<20} {reward_str:>15} {stats.get('completion',0):>7.1f}% {stats.get('on_time_rate',100):>7.1f}% "
              f"{stats.get('avg_latency',0):>9.1f} {stats.get('battery_used_pct',0):>9.1f}% {action_str}")
    
    print("\n" + "="*80)
    
    # Find best algorithm
    all_algs = {**marl_results}
    if baseline_results:
        for name, collector in baseline_results.items():
            if hasattr(collector, 'summary'):
                all_algs[name] = collector.summary()
    
    best_reward = max(all_algs.items(), key=lambda x: x[1].get('reward_mean', float('-inf')))
    print(f"\n🏆 Best by Reward: {best_reward[0]} ({best_reward[1].get('reward_mean',0):.1f})")


def save_comprehensive_checkpoint(output_dir: str, marl_results: Dict, 
                                   baseline_results: Dict = None, config: Dict = None):
    """
    Save all results with comprehensive metrics for later analysis.
    
    Directory structure:
    output_dir/
    ├── config.json
    ├── summary_table.csv           # Human-readable summary
    ├── summary_table.tex           # LaTeX table
    ├── aggregate_metrics.csv       # Machine-readable metrics
    ├── figures/
    │   └── *.png
    ├── baselines/
    │   └── {policy}/episode_metrics.csv
    └── marl/
        └── {algorithm}/
            ├── summary.json
            └── episode_metrics.csv
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    if config:
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
    
    # Save summary table
    summary_df = create_summary_table(marl_results, baseline_results, 
                                       str(path / "summary_table.csv"))
    
    # Save LaTeX table
    create_latex_table(marl_results, baseline_results, 
                       str(path / "summary_table.tex"))
    
    # Save aggregate metrics (machine-readable)
    agg_df = aggregate_marl_results(marl_results)
    agg_df.to_csv(path / "aggregate_metrics.csv", index=False)
    
    # Save baseline episode metrics
    if baseline_results:
        baseline_path = path / "baselines"
        baseline_path.mkdir(exist_ok=True)
        for name, collector in baseline_results.items():
            policy_path = baseline_path / name.replace(" ", "_").replace("-", "_")
            policy_path.mkdir(exist_ok=True)
            if hasattr(collector, 'to_df'):
                collector.to_df().to_csv(policy_path / "episode_metrics.csv", index=False)
    
    # Save MARL results
    marl_path = path / "marl"
    marl_path.mkdir(exist_ok=True)
    for alg, stats in marl_results.items():
        alg_path = marl_path / alg
        alg_path.mkdir(exist_ok=True)
        
        # Summary JSON
        summary = {k: v for k, v in stats.items() 
                   if not isinstance(v, (list, np.ndarray)) or k in ['rewards']}
        with open(alg_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        # Episode metrics if available
        if 'rewards' in stats:
            ep_data = {'episode': list(range(len(stats['rewards']))),
                       'reward': stats['rewards']}
            if 'completions' in stats:
                ep_data['completion'] = stats['completions']
            if 'on_time_rates_list' in stats:
                ep_data['on_time_rate'] = stats['on_time_rates_list']
            if 'ep_mean_latencies' in stats:
                ep_data['mean_latency'] = stats['ep_mean_latencies']
            pd.DataFrame(ep_data).to_csv(alg_path / "episode_metrics.csv", index=False)
    
    # Generate figures
    generate_comprehensive_figures(marl_results, baseline_results, str(path / "figures"))
    
    # Print summary
    print_results_summary(marl_results, baseline_results)
    
    print(f"\n✓ Comprehensive checkpoint saved to {output_dir}")
    return summary_df


# =============================================================================
# REPLICATION CHECKLIST
# =============================================================================

REPLICATION_CHECKLIST = """
================================================================================
REPLICATION CHECKLIST
================================================================================

Required files for reproducing results:

1. CODE
   ├── fieldvision_complete.py     # Simulation + Training
   └── evaluation_pipeline.py      # This file (baselines + metrics)

2. CONFIG
   └── config.json                 # All experiment parameters

3. DATA
   ├── baseline_results/           # Per-policy evaluation
   │   └── {policy}/episode_metrics.csv
   ├── marl_results/               # Trained agent evaluation
   │   └── {algorithm}/
   │       ├── episode_metrics.csv
   │       └── training_history.json
   └── summary.csv                 # Aggregated comparison

4. MODELS (optional)
   └── models/
       ├── {algorithm}_actor.pt
       └── {algorithm}_critic.pt

5. FIGURES
   └── figures/
       ├── reward_comparison.png
       ├── action_distribution.png
       └── completion_rate.png

================================================================================
"""

def print_checklist():
    print(REPLICATION_CHECKLIST)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Evaluation Pipeline - Testing...")
    
    # Quick test
    try:
        from fieldvision_complete import make_multi_drone_env, MAPPOWrapper, get_difficulty_config
        
        config = get_difficulty_config("streaming_hard")
        def make_env(s):
            return MAPPOWrapper(make_multi_drone_env(seed=s, **config))
        
        # Test one policy
        policy = LocalPolicy()
        collector = evaluate_policy(policy, make_env, n_episodes=3, max_steps=100)
        print(f"Local-Only: {collector.summary()}")
        
        print("\nAll tests passed!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Place fieldvision_complete.py in the same directory")
