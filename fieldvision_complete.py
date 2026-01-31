# -*- coding: utf-8 -*-
"""
FieldVision Complete - Simulation + MARL Training + Difficulty Configs
=======================================================================

COMPLETE file for running experiments. Includes:
1. Multi-drone offload simulator (OffloadSim)
2. Gymnasium environment (MultiDroneOffloadEnv)
3. MAPPO wrapper for multi-agent RL
4. Difficulty-focused configurations
5. MARL training (Centralized PPO, MAPPO, MAPO)

Usage:
    !pip install gymnasium torch matplotlib pandas
    from fieldvision_complete import *
    results = run_comparison_experiment(total_timesteps=200_000, difficulty="streaming_hard")
"""

from __future__ import annotations
import os, time, json, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Iterable
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# DATA STRUCTURES
# =============================================================================

AgentName = str
Action = int

@dataclass
class Chunk:
    size_mb: float
    task_id: str
    initial_deadline: int
    deadline_remaining: int
    priority: float
    mosaic_id: int
    type: str
    source_drone: str
    created_timestep: int = 0  # When this chunk was created/arrived

@dataclass
class NetworkState:
    drone_ground_latency_ms: float
    drone_ground_bandwidth_mbps: float
    drone_ground_connected: bool
    ground_cloud_latency_ms: float
    ground_cloud_bandwidth_mbps: float
    ground_cloud_connected: bool
    drone_ground_loss_rate: float = 0.0

@dataclass
class AgentRuntime:
    name: AgentName
    max_load: float
    speed: float
    battery_drain: float
    battery_level: float
    parallelism: int
    cost_per_task: float
    allowed_types: Iterable[str]
    current_load: float = 0.0
    active_tasks: int = 0
    is_available: bool = True
    
    def copy(self):
        return AgentRuntime(
            self.name, self.max_load, self.speed, self.battery_drain, self.battery_level,
            self.parallelism, self.cost_per_task, list(self.allowed_types),
            self.current_load, self.active_tasks, self.is_available
        )

@dataclass
class InFlightTransfer:
    chunk: Chunk
    target_agent: str
    source_drone: str
    total_size_mb: float
    transferred_mb: float = 0.0
    started_timestep: int = 0
    cumulative_latency_ms: float = 0.0
    timesteps_in_flight: int = 0
    failed: bool = False

# =============================================================================
# NETWORK PROVIDER
# =============================================================================

class SinusoidalNetProvider:
    def __init__(self, wavelength=50.0, amplitude=4.0, baseline=5.0, noise_std=0.5,
                 phase=0.0, latency_base=30.0, latency_scale=50.0, seed=None):
        self.wavelength, self.amplitude, self.baseline = wavelength, amplitude, baseline
        self.noise_std, self.phase = noise_std, phase
        self.latency_base, self.latency_scale = latency_base, latency_scale
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

    def reset(self):
        self.timestep = 0

    def next(self):
        angular_freq = 2 * np.pi / self.wavelength
        raw_bw = self.baseline + self.amplitude * np.sin(angular_freq * self.timestep + self.phase)
        noisy_bw = max(0.0, raw_bw + self.rng.normal(0, self.noise_std))
        connected = noisy_bw > 0.1
        
        if noisy_bw > 0.1:
            max_bw = self.baseline + self.amplitude
            latency = self.latency_base + self.latency_scale * (1.0 - noisy_bw / max_bw)
            latency = max(5.0, latency + self.rng.normal(0, 10))
            loss_rate = max(0.0, 0.05 * (1.0 - noisy_bw / max_bw))
        else:
            latency, loss_rate = 2000.0, 0.3
        
        self.timestep += 1
        return {"drone_ground_latency_ms": latency, "drone_ground_bandwidth_mbps": noisy_bw,
                "drone_ground_connected": connected, "drone_ground_loss_rate": loss_rate}

# =============================================================================
# DIFFICULTY CONFIGURATIONS
# =============================================================================

def get_difficulty_config(difficulty: str = "streaming_hard") -> dict:
    """Get difficulty-focused environment configuration.
    
    Network parameters explained:
    - base_baseline: Average bandwidth in Mbps (lower = worse connectivity)
    - base_amplitude: How much bandwidth varies (higher = more variable)
    - base_wavelength: Period of bandwidth cycle in timesteps (shorter = faster changes)
    - network_noise_std: Random noise on bandwidth (higher = more unpredictable)
    
    Agricultural scenarios:
    - rural_field: Typical rural farm with limited connectivity
    - variable_terrain: Hilly terrain causing signal variations
    - weather_stress: Poor conditions (wind, interference)
    - peak_survey: Heavy workload during harvest imagery collection
    """
    configs = {
        # === Standard difficulty progression ===
        "easy": {"chunks_per_drone": [6,6,6,6], "chunk_size_mb": 8.0, "base_deadline": 40,
                 "base_baseline": 8.0, "base_amplitude": 3.0, "base_wavelength": 40.0,
                 "network_noise_std": 0.5, "max_steps": 150, "streaming_enabled": False},
        "medium": {"chunks_per_drone": [10,10,10,10], "chunk_size_mb": 10.0, "base_deadline": 25,
                   "base_baseline": 6.0, "base_amplitude": 4.0, "base_wavelength": 30.0,
                   "network_noise_std": 0.8, "max_steps": 200, "streaming_enabled": True, "stream_interval": 5},
        "hard": {"chunks_per_drone": [12,12,12,12], "chunk_size_mb": 12.0, "base_deadline": 20,
                 "base_baseline": 5.5, "base_amplitude": 4.0, "base_wavelength": 25.0,
                 "network_noise_std": 1.0, "max_steps": 300, "streaming_enabled": True, "stream_interval": 3},
        "streaming_hard": {"chunks_per_drone": [12,12,12,12], "chunk_size_mb": 14.0, "base_deadline": 18,
                          "base_baseline": 5.5, "base_amplitude": 4.0, "base_wavelength": 20.0,
                          "network_noise_std": 1.2, "max_steps": 300, "streaming_enabled": True, "stream_interval": 3},
        "stress": {"chunks_per_drone": [15,15,15,15], "chunk_size_mb": 14.0, "base_deadline": 16,
                   "base_baseline": 5.0, "base_amplitude": 4.5, "base_wavelength": 15.0,
                   "network_noise_std": 1.5, "max_steps": 400, "streaming_enabled": True, "stream_interval": 2},
        
        # === Realistic Agricultural Scenarios ===
        
        # Rural field: Low bandwidth typical of remote farms, moderate reliability
        # Network often too slow for cloud, edge GPU is critical
        "rural_field": {
            "chunks_per_drone": [10,10,10,10], "chunk_size_mb": 12.0, "base_deadline": 25,
            "base_baseline": 4.0,  # Low bandwidth (rural 4G/LTE)
            "base_amplitude": 2.5, # Moderate variation
            "base_wavelength": 35.0,
            "network_noise_std": 1.0,
            "max_steps": 250, "streaming_enabled": True, "stream_interval": 4
        },
        
        # Variable terrain: Hilly farmland with signal shadows
        # Bandwidth swings dramatically as drones move through terrain
        "variable_terrain": {
            "chunks_per_drone": [12,12,12,12], "chunk_size_mb": 12.0, "base_deadline": 22,
            "base_baseline": 5.0,
            "base_amplitude": 4.5,  # HIGH variation (signal shadows from hills)
            "base_wavelength": 15.0,  # Faster changes (moving through terrain)
            "network_noise_std": 1.5,  # Unpredictable due to multipath
            "max_steps": 300, "streaming_enabled": True, "stream_interval": 3
        },
        
        # Weather stress: Poor conditions causing interference
        # High noise, frequent packet loss, agents must adapt quickly
        "weather_stress": {
            "chunks_per_drone": [10,10,10,10], "chunk_size_mb": 10.0, "base_deadline": 30,
            "base_baseline": 4.5,
            "base_amplitude": 3.5,
            "base_wavelength": 25.0,
            "network_noise_std": 2.5,  # VERY HIGH noise (weather interference)
            "max_steps": 250, "streaming_enabled": True, "stream_interval": 4
        },
        
        # Peak survey: Harvest time - lots of imagery, tight deadlines
        # Heavy workload, network contention, time pressure
        "peak_survey": {
            "chunks_per_drone": [18,18,18,18], "chunk_size_mb": 16.0, "base_deadline": 15,
            "base_baseline": 6.0,
            "base_amplitude": 3.5,
            "base_wavelength": 20.0,
            "network_noise_std": 1.2,
            "max_steps": 400, "streaming_enabled": True, "stream_interval": 2  # Fast streaming
        },
        
        # Network-aware challenge: Designed to require adaptive network decisions
        # Cloud has high latency, edge varies dramatically, local is slow
        "network_aware": {
            "chunks_per_drone": [12,12,12,12], "chunk_size_mb": 14.0, "base_deadline": 20,
            "base_baseline": 5.0,  # Moderate baseline
            "base_amplitude": 4.0,  # Significant swings
            "base_wavelength": 12.0,  # Fast cycling (requires quick adaptation)
            "network_noise_std": 2.0,  # High unpredictability
            "max_steps": 300, "streaming_enabled": True, "stream_interval": 3
        },
    }
    if difficulty not in configs:
        raise ValueError(f"Unknown difficulty: {difficulty}. Available: {list(configs.keys())}")
    return {**configs[difficulty], "workload_band": None}

get_env_config = get_difficulty_config

# =============================================================================
# OFFLOAD SIMULATOR
# =============================================================================

class OffloadSim:
    DECISION_AGENT_NAMES = ["MPD_0", "MPD_1", "LPD_0", "LPD_1"]

    def __init__(self, mpd_spec, lpd_spec, gpu_spec, cloud_spec, chunks_per_drone,
                 chunk_size_mb=10.0, base_deadline=50, base_priority=0.7, workload_band=None,
                 streaming_enabled=False, stream_interval=5, base_wavelength=50.0,
                 base_amplitude=3.0, base_baseline=8.0, network_noise_std=0.3,
                 ground_cloud_latency_ms=90.0, ground_cloud_bandwidth_mbps=200.0,
                 max_steps=100, base_process_reward=10.0, offload_initiated_reward=8.0,
                 local_mpd_bonus=1.0, local_lpd_bonus=2.0, cannot_process_penalty=1.0,
                 cannot_offload_penalty=1.0, pending_per_chunk_penalty=0.1,
                 delay_penalty_scale=0.5, delay_penalty_cap=5.0, priority_weight=2.0,
                 deadline_progress_weight=0.1, battery_penalty_scale=0.5,
                 battery_empty_penalty=10.0, deadline_miss_penalty=30.0,
                 transfer_failure_penalty=15.0, transfer_latency_penalty_scale=0.001,
                 mosaic_bonus=40.0, seed=None, **kwargs):
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._seed = seed
        
        self.params = {
            "chunk_size_mb": chunk_size_mb, "base_deadline": base_deadline,
            "base_priority": base_priority, "max_steps": max_steps,
            "base_process_reward": base_process_reward, "offload_initiated_reward": offload_initiated_reward,
            "local_mpd_bonus": local_mpd_bonus, "local_lpd_bonus": local_lpd_bonus,
            "cannot_process_penalty": cannot_process_penalty, "cannot_offload_penalty": cannot_offload_penalty,
            "pending_per_chunk_penalty": pending_per_chunk_penalty, "delay_penalty_scale": delay_penalty_scale,
            "delay_penalty_cap": delay_penalty_cap, "priority_weight": priority_weight,
            "deadline_progress_weight": deadline_progress_weight, "battery_penalty_scale": battery_penalty_scale,
            "battery_empty_penalty": battery_empty_penalty, "deadline_miss_penalty": deadline_miss_penalty,
            "transfer_failure_penalty": transfer_failure_penalty, "transfer_latency_penalty_scale": transfer_latency_penalty_scale,
            "mosaic_bonus": mosaic_bonus,
            "workload_band": workload_band, "streaming_enabled": streaming_enabled, "stream_interval": stream_interval,
        }
        
        self._net_params = {"base_wavelength": base_wavelength, "base_amplitude": base_amplitude,
                           "base_baseline": base_baseline, "noise_std": network_noise_std}
        self.ground_cloud_latency_ms = ground_cloud_latency_ms
        self.ground_cloud_bandwidth_mbps = ground_cloud_bandwidth_mbps
        self.ground_cloud_connected = True
        
        self._mpd_template, self._lpd_template = mpd_spec, lpd_spec
        self._gpu_template, self._cloud_template = gpu_spec, cloud_spec
        self._base_chunks_per_drone = chunks_per_drone
        
        self.decision_agent_names = self.DECISION_AGENT_NAMES
        self.max_steps = max_steps
        self.time_step = 0
        
        self.compute_agents = {}
        self.task_queues = {}
        self.drone_networks = {}
        self.drone_network_states = {}
        self.in_flight_transfers = {}
        self._transfer_counter = 0
        self.mosaic_task_map = {}
        self.processed_chunks_count = 0
        self.total_chunks = 0
        self.deadline_misses = 0
        self.transfer_failures = 0
        self._pending_chunks = {}
        self._next_stream_step = {}

    def reset(self, seed=None):
        """Reset the simulation. Optionally provide a new seed for variety."""
        if seed is not None:
            self._seed = seed
            self.rng = np.random.RandomState(seed)
        
        self.time_step = 0
        self._transfer_counter = 0
        self.processed_chunks_count = 0
        self.deadline_misses = 0  # Mosaic-level deadline misses
        self.transfer_failures = 0
        self.tasks_on_time = 0  # Tasks completed before their individual deadline
        self.tasks_late = 0     # Tasks completed after their individual deadline
        
        self.compute_agents = {
            "MPD_0": self._mpd_template.copy(), "MPD_1": self._mpd_template.copy(),
            "LPD_0": self._lpd_template.copy(), "LPD_1": self._lpd_template.copy(),
            "Desktop GPU": self._gpu_template.copy(), "Cloud": self._cloud_template.copy(),
        }
        for name, agent in self.compute_agents.items():
            agent.name = name
            agent.battery_level = 1.0
            agent.current_load = 0.0
            agent.active_tasks = 0
        
        chunks_per_drone = self._base_chunks_per_drone
        if self.params.get("workload_band"):
            band = self.params["workload_band"]
            if isinstance(band, tuple):
                total = self.rng.randint(band[0], band[1])
            elif band == "small": total = self.rng.randint(8, 16)
            elif band == "mixed": total = self.rng.randint(16, 32)
            elif band == "high": total = self.rng.randint(32, 48)
            else: total = sum(chunks_per_drone)
            base = total // 4
            chunks_per_drone = [base + (1 if i < total % 4 else 0) for i in range(4)]
        
        self.task_queues = {name: deque() for name in self.decision_agent_names}
        self.mosaic_task_map = {}
        self._pending_chunks = {name: [] for name in self.decision_agent_names}
        self._next_stream_step = {name: 0 for name in self.decision_agent_names}
        
        streaming = self.params.get("streaming_enabled", False)
        mosaic_id = 0
        for drone_idx, (drone_name, n_chunks) in enumerate(zip(self.decision_agent_names, chunks_per_drone)):
            for chunk_idx in range(n_chunks):
                # created_timestep=0 for initial chunks, will be updated when streamed
                chunk = Chunk(self.params["chunk_size_mb"], f"{drone_name}_task_{chunk_idx}",
                            self.params["base_deadline"], self.params["base_deadline"],
                            self.params.get("base_priority", 0.7), mosaic_id, "any", drone_name,
                            created_timestep=0)
                if streaming and chunk_idx > 0:
                    self._pending_chunks[drone_name].append(chunk)
                else:
                    self.task_queues[drone_name].append(chunk)
            self.mosaic_task_map[mosaic_id] = {"remaining": n_chunks, "deadline": self.params["base_deadline"] + 10,
                                               "completed": False, "chunks_done": []}
            mosaic_id += 1
        self.total_chunks = sum(chunks_per_drone)
        
        self.drone_networks = {}
        phases = [0, np.pi/2, np.pi, 3*np.pi/2]
        for idx, name in enumerate(self.decision_agent_names):
            self.drone_networks[name] = SinusoidalNetProvider(
                self._net_params["base_wavelength"], self._net_params["base_amplitude"],
                self._net_params["base_baseline"], self._net_params["noise_std"],
                phases[idx], seed=self._seed + idx if self._seed else None)
        
        self._update_networks()
        self.in_flight_transfers = {}
        
        # Return observation dict (Gymnasium convention)
        return self.get_obs(), {}

    def _update_networks(self):
        self.drone_network_states = {}
        for name, provider in self.drone_networks.items():
            net = provider.next()
            self.drone_network_states[name] = NetworkState(
                net["drone_ground_latency_ms"], net["drone_ground_bandwidth_mbps"],
                net["drone_ground_connected"], self.ground_cloud_latency_ms,
                self.ground_cloud_bandwidth_mbps, self.ground_cloud_connected,
                net.get("drone_ground_loss_rate", 0.0))
        self.ground_cloud_bandwidth_mbps = float(np.clip(self.np_rng.normal(200, 20), 100, 300))
        self.ground_cloud_latency_ms = float(np.clip(self.np_rng.normal(90, 10), 50, 150))

    def begin_step(self):
        """Phase 1: Advance time, networks, streaming. Returns which agents have tasks."""
        self.time_step += 1
        self._update_networks()
        
        if self.params.get("streaming_enabled"):
            interval = self.params.get("stream_interval", 5)
            for name in self.decision_agent_names:
                if self.time_step >= self._next_stream_step[name] and self._pending_chunks[name]:
                    chunk = self._pending_chunks[name].pop(0)
                    chunk.created_timestep = self.time_step
                    self.task_queues[name].append(chunk)
                    self._next_stream_step[name] = self.time_step + interval
        
        # Return which agents have tasks (AFTER streaming)
        return {name: len(self.task_queues[name]) > 0 for name in self.decision_agent_names}
    
    def complete_step(self, actions):
        """Phase 2: Process actions and finalize step. Call after begin_step()."""
        team_reward = 0.0
        agent_rewards = {name: 0.0 for name in self.decision_agent_names}
        info = {"action_results": {}, "transfers_completed": 0, "transfers_failed": 0,
                "completed_task_latencies": []}
        
        # Track which agents had tasks to process
        agents_with_tasks = {name: len(self.task_queues[name]) > 0 for name in self.decision_agent_names}
        
        for drone_name, action in actions.items():
            r, result, task_latency = self._process_action(drone_name, action)
            team_reward += r
            agent_rewards[drone_name] += r
            info["action_results"][drone_name] = result
            if task_latency is not None:
                info["completed_task_latencies"].append(task_latency)
        
        transfer_reward, completed, failed, transfer_task_latencies, transfer_sources = self._advance_transfers()
        team_reward += transfer_reward
        for source_drone, t_reward in transfer_sources:
            agent_rewards[source_drone] += t_reward
        info["transfers_completed"] = completed
        info["transfers_failed"] = failed
        info["completed_task_latencies"].extend(transfer_task_latencies)
        self.transfer_failures += failed
        
        self._advance_compute()
        self._decrement_deadlines()
        
        # Per-agent penalties
        for name in self.decision_agent_names:
            queue_penalty = len(self.task_queues[name]) * self.params["pending_per_chunk_penalty"]
            agent_rewards[name] -= queue_penalty
            team_reward -= queue_penalty
            
            drone = self.compute_agents[name]
            battery_penalty = (1.0 - drone.battery_level) * self.params["battery_penalty_scale"]
            agent_rewards[name] -= battery_penalty
            team_reward -= battery_penalty
            if drone.battery_level <= 0.01:
                agent_rewards[name] -= self.params["battery_empty_penalty"]
                team_reward -= self.params["battery_empty_penalty"]
        
        # Mosaic completion/deadline logic
        for mid, m in self.mosaic_task_map.items():
            if not m["completed"] and m["remaining"] <= 0:
                bonus = self.params["mosaic_bonus"]
                team_reward += bonus
                per_agent_bonus = bonus / len(self.decision_agent_names)
                for name in self.decision_agent_names:
                    agent_rewards[name] += per_agent_bonus
                m["completed"] = True
            if not m.get("deadline_missed", False) and m["deadline"] <= self.time_step and m["remaining"] > 0:
                penalty = self.params["deadline_miss_penalty"]
                team_reward -= penalty
                per_agent_penalty = penalty / len(self.decision_agent_names)
                for name in self.decision_agent_names:
                    agent_rewards[name] -= per_agent_penalty
                m["deadline_missed"] = True
                self.deadline_misses += 1
        
        info["processed_chunks"] = self.processed_chunks_count
        info["total_chunks"] = self.total_chunks
        info["deadline_misses"] = self.deadline_misses  # Mosaic-level
        info["transfer_failures"] = self.transfer_failures
        info["tasks_on_time"] = self.tasks_on_time  # Per-task: completed before deadline
        info["tasks_late"] = self.tasks_late        # Per-task: completed after deadline
        info["agent_rewards"] = agent_rewards
        info["agents_with_tasks"] = agents_with_tasks  # For training filtering
        
        # Battery levels at end of step (for tracking depletion)
        info["battery_levels"] = {
            name: self.compute_agents[name].battery_level 
            for name in self.decision_agent_names
        }
        
        return team_reward, info

    def step(self, actions):
        """Execute one step. Returns Gymnasium-style (obs, reward, terminated, truncated, info)."""
        self.begin_step()
        reward, info = self.complete_step(actions)
        
        # Check termination
        terminated = self._check_done()
        truncated = self.time_step >= self.max_steps
        
        # Get observation
        obs = self.get_obs()
        
        return obs, reward, terminated, truncated, info

    def _priority_deadline_reward(self, chunk):
        """Calculate priority and deadline progress reward."""
        w_p = self.params["priority_weight"]
        w_d = self.params["deadline_progress_weight"]
        progressed = max(0, chunk.initial_deadline - chunk.deadline_remaining)
        return (chunk.priority * w_p) + (progressed * w_d)

    def _process_action(self, drone_name, action):
        queue = self.task_queues[drone_name]
        if not queue:
            return 0.0, {"action": action, "success": False, "reason": "empty_queue"}, None
        
        chunk = queue[0]
        drone_net = self.drone_network_states[drone_name]
        reward = 0.0
        result = {"action": action, "success": False, "reason": None}
        task_latency = None  # Latency from task creation to completion
        
        if action == 0:
            drone = self.compute_agents[drone_name]
            if drone.active_tasks < drone.parallelism and drone.battery_level > 0.01:
                queue.popleft()
                drone.current_load += chunk.size_mb
                drone.active_tasks = min(drone.parallelism, drone.active_tasks + 1)
                self.processed_chunks_count += 1
                self.mosaic_task_map[chunk.mosaic_id]["remaining"] -= 1
                
                # Track per-task deadline status
                if chunk.deadline_remaining > 0:
                    self.tasks_on_time += 1
                else:
                    self.tasks_late += 1
                
                # Calculate task latency (timesteps from creation to completion)
                task_latency = self.time_step - chunk.created_timestep
                
                # EFFICIENCY-BASED REWARD with stronger contrast
                # MPD: speed=20, LPD: speed=5
                # We want LPDs to clearly see Local is worse
                eff_speed = max(1e-6, drone.speed * max(1, drone.parallelism))
                processing_time = chunk.size_mb / eff_speed
                
                # Stronger efficiency factor: quadratic penalty for slowness
                # MPD: 10/20 = 0.5 -> (1 - 0.5^2) = 0.75 -> reward = base * 0.75
                # LPD: 10/5 = 2.0 -> capped at 1.0 -> (1 - 1.0) = 0.0 -> reward = base * 0.3
                normalized_time = min(1.0, processing_time / 2.0)  # Normalize to [0, 1]
                efficiency_factor = 1.0 - normalized_time ** 2  # Quadratic penalty
                reward += self.params["base_process_reward"] * (0.3 + 0.7 * efficiency_factor)
                
                # Extra penalty for slow drones (LPDs)
                if drone.speed < 10:  # LPD threshold
                    reward -= 2.0  # Explicit "you're slow, should offload" penalty
                
                # Priority/deadline reward
                reward += self._priority_deadline_reward(chunk)
                
                # Delay penalty
                delay_penalty = min(self.params["delay_penalty_cap"], processing_time * self.params["delay_penalty_scale"])
                reward -= delay_penalty
                
                result["success"] = True
            else:
                reward -= self.params["cannot_process_penalty"]
                result["reason"] = "capacity"
        
        elif action in [1, 2]:
            target = "Desktop GPU" if action == 1 else "Cloud"
            agent = self.compute_agents[target]
            if agent.active_tasks < agent.parallelism and drone_net.drone_ground_connected:
                queue.popleft()
                tid = f"transfer_{self._transfer_counter}"
                self._transfer_counter += 1
                self.in_flight_transfers[tid] = InFlightTransfer(
                    chunk, target, drone_name, chunk.size_mb, started_timestep=self.time_step)
                
                # Give ALL offload reward IMMEDIATELY (was 70/30 split)
                # This fixes credit assignment - agent sees full benefit of offloading right away
                reward += self.params["offload_initiated_reward"]
                
                # Priority reward also immediate
                reward += self._priority_deadline_reward(chunk)
                
                if target == "Cloud":
                    reward -= agent.cost_per_task * 0.5
                result["success"] = True
            else:
                reward -= self.params["cannot_offload_penalty"]
                result["reason"] = "full_or_disconnected"
        
        return reward, result, task_latency

    def _advance_transfers(self):
        reward, completed, failed = 0.0, 0, 0
        task_latencies = []
        transfer_sources = []  # (source_drone, reward) pairs for credit assignment
        to_remove = []
        
        for tid, t in self.in_flight_transfers.items():
            t.timesteps_in_flight += 1
            net = self.drone_network_states[t.source_drone]
            
            if t.target_agent == "Cloud":
                connected = net.drone_ground_connected and self.ground_cloud_connected
                bw = min(net.drone_ground_bandwidth_mbps, self.ground_cloud_bandwidth_mbps)
            else:
                connected = net.drone_ground_connected
                bw = net.drone_ground_bandwidth_mbps
            
            if not connected:
                t.failed = True
                to_remove.append(tid)
                penalty = self.params["transfer_failure_penalty"]
                reward -= penalty
                transfer_sources.append((t.source_drone, -penalty))  # Credit failure to source
                failed += 1
                continue
            
            t.transferred_mb += bw / 8.0
            if t.transferred_mb >= t.total_size_mb:
                to_remove.append(tid)
                completed += 1
                
                task_latency = self.time_step - t.chunk.created_timestep
                task_latencies.append(task_latency)
                
                self.processed_chunks_count += 1
                
                # Track per-task deadline status
                if t.chunk.deadline_remaining > 0:
                    self.tasks_on_time += 1
                else:
                    self.tasks_late += 1
                
                target = self.compute_agents[t.target_agent]
                target.current_load += t.chunk.size_mb
                target.active_tasks = min(target.parallelism, target.active_tasks + 1)
                self.mosaic_task_map[t.chunk.mosaic_id]["remaining"] -= 1
                
                # Cloud cost - attribute to source drone
                if t.target_agent == "Cloud":
                    cost = self.compute_agents["Cloud"].cost_per_task * 0.5
                    reward -= cost
                    transfer_sources.append((t.source_drone, -cost))
        
        for tid in to_remove:
            del self.in_flight_transfers[tid]
        return reward, completed, failed, task_latencies, transfer_sources

    def _advance_compute(self):
        for agent in self.compute_agents.values():
            if agent.current_load > 0:
                processed = min(agent.current_load, agent.speed * max(1, agent.parallelism))
                agent.current_load -= processed
                if agent.battery_drain > 0:
                    agent.battery_level = max(0, agent.battery_level - agent.battery_drain * processed)
                if agent.current_load <= 1e-6:
                    agent.current_load = 0
                    agent.active_tasks = 0
            else:
                agent.active_tasks = 0

    def _decrement_deadlines(self):
        """Decrement deadline for ALL chunks in ALL queues - they all tick down every timestep."""
        for q in self.task_queues.values():
            for chunk in q:
                chunk.deadline_remaining = max(0, chunk.deadline_remaining - 1)

    def _check_done(self):
        return (all(len(q) == 0 for q in self.task_queues.values()) and
                all(len(p) == 0 for p in self._pending_chunks.values()) and
                len(self.in_flight_transfers) == 0)

    def get_obs(self):
        """Build per-agent observations with proper local/global separation.
        
        Local observation (per drone) - 16 features:
        - Queue state (4): length, deadline_ratio, priority, pending
        - Drone state (4): battery, load, active_tasks, is_available
        - Drone capabilities (2): speed, battery_drain - for policy differentiation
        - Network (4): bandwidth, latency, connected, loss_rate
        - Resource availability (2): GPU busy, Cloud busy - for coordination
        
        Global observation (for centralized critic only):
        - All local observations concatenated (16 * 4 = 64)
        - GPU/Cloud state (4)
        - In-flight transfer counts (2)
        - Global timestep (1)
        
        Total: 71 features (16*4 + 7)
        """
        local_obs = {}
        
        # Get GPU/Cloud state for resource availability features
        gpu = self.compute_agents["Desktop GPU"]
        cloud = self.compute_agents["Cloud"]
        
        # Build LOCAL observations
        for name in self.decision_agent_names:
            drone = self.compute_agents[name]
            net = self.drone_network_states.get(name, NetworkState(100, 5, True, 90, 200, True, 0))
            queue = self.task_queues[name]
            chunk = queue[0] if queue else None
            pending = len(self._pending_chunks.get(name, []))
            
            # LOCAL STATE (16 features)
            local = [
                # Queue state (4)
                len(queue) / 10.0,
                (chunk.deadline_remaining / max(1, chunk.initial_deadline)) if chunk else 0,
                chunk.priority if chunk else 0,
                pending / 10.0,  # Pending chunks for this drone
                
                # Own drone state (4)
                drone.battery_level,
                drone.current_load / max(1, drone.max_load),
                drone.active_tasks / max(1, drone.parallelism),
                float(drone.is_available),
                
                # Drone capabilities (2) - CRITICAL for learning different policies!
                drone.speed / 20.0,  # Normalized by max speed (MPD=1.0, LPD=0.25)
                drone.battery_drain * 100,  # MPD=0.5, LPD=1.0
                
                # Own network conditions (4)
                net.drone_ground_bandwidth_mbps / 15.0,
                net.drone_ground_latency_ms / 200.0,
                float(net.drone_ground_connected),
                net.drone_ground_loss_rate,
                
                # Shared resource availability (2) - drone can query ground station
                gpu.active_tasks / max(1, gpu.parallelism),  # GPU busy ratio (0=free, 1=full)
                cloud.active_tasks / max(1, cloud.parallelism),  # Cloud busy ratio
            ]
            
            local_obs[name] = np.array(local, dtype=np.float32)
        
        return local_obs
    
    def get_global_obs(self):
        """Build global observation for centralized critic.
        
        Contains full system state that individual drones cannot observe.
        """
        gpu = self.compute_agents["Desktop GPU"]
        cloud = self.compute_agents["Cloud"]
        
        in_flight_gpu = sum(1 for t in self.in_flight_transfers.values() 
                          if t.target_agent == "Desktop GPU")
        in_flight_cloud = sum(1 for t in self.in_flight_transfers.values() 
                            if t.target_agent == "Cloud")
        
        # Global state (7 features)
        global_state = [
            self.time_step / self.max_steps,
            gpu.current_load / max(1, gpu.max_load),
            gpu.active_tasks / max(1, gpu.parallelism),
            cloud.current_load / max(1, cloud.max_load),
            cloud.active_tasks / max(1, cloud.parallelism),
            in_flight_gpu / 10.0,
            in_flight_cloud / 10.0,
        ]
        
        # Concatenate all local observations + global state
        local_obs = self.get_obs()
        all_local = np.concatenate([local_obs[n] for n in self.decision_agent_names])
        
        return np.concatenate([all_local, np.array(global_state, dtype=np.float32)])

# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class MultiDroneOffloadEnv(gym.Env):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.decision_agents = sim.decision_agent_names
        self.n_agents = len(self.decision_agents)
        self.action_space = spaces.Dict({name: spaces.Discrete(3) for name in self.decision_agents})
        # Local observation: 16 features per drone (added speed, battery_drain, GPU/Cloud availability)
        self.observation_space = spaces.Dict({
            name: spaces.Box(-np.inf, np.inf, (16,), np.float32) for name in self.decision_agents})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)  # Pass seed to sim for scenario variety
        return self.sim.get_obs(), {}

    def step(self, actions):
        # OffloadSim.step now returns Gymnasium format directly
        return self.sim.step(actions)

class MAPPOWrapper:
    """Wrapper providing separate local/global observations for MAPPO.
    
    Local obs: What each drone can observe (16 features)
      - Queue state (4): length, deadline_ratio, priority, pending
      - Drone state (4): battery, load_ratio, task_ratio, is_available  
      - Drone capabilities (2): speed, battery_drain - for policy differentiation
      - Network (4): bandwidth, latency, connected, loss_rate
      - Resource availability (2): GPU busy ratio, Cloud busy ratio - for coordination
      
    Global obs: Full system state for centralized critic (16*4 + 7 = 71 features)
    """
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.n_agents = env.n_agents
        self.agent_names = env.decision_agents
        self.n_actions = 3
        
        # Get actual observation dimensions
        self.env.reset()
        local_obs = self.sim.get_obs()
        global_obs = self.sim.get_global_obs()
        
        self.local_obs_dim = local_obs[self.agent_names[0]].shape[0]  # 12
        self.global_obs_dim = global_obs.shape[0]  # 55

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        local_obs_dict = self.sim.get_obs()
        local_obs = np.stack([local_obs_dict[n] for n in self.agent_names])
        global_obs = self.sim.get_global_obs()
        return local_obs, global_obs, {}

    def begin_step(self):
        """Phase 1: Advance time, networks, streaming. Returns which agents have tasks."""
        agents_with_tasks = self.sim.begin_step()
        # Return as array for easy indexing
        return np.array([agents_with_tasks[n] for n in self.agent_names], dtype=bool)
    
    def complete_step(self, actions):
        """Phase 2: Process actions and finalize. Call after begin_step()."""
        # Handle both array and dict input
        if isinstance(actions, dict):
            action_dict = {n: int(actions[n]) for n in self.agent_names}
        else:
            action_dict = {n: int(actions[i]) for i, n in enumerate(self.agent_names)}
        
        reward, info = self.sim.complete_step(action_dict)
        
        # Check termination
        term = self.sim._check_done()
        trunc = self.sim.time_step >= self.sim.max_steps
        
        # Get observations
        local_obs_dict = self.sim.get_obs()
        local_obs = np.stack([local_obs_dict[n] for n in self.agent_names])
        global_obs = self.sim.get_global_obs()
        
        # Per-agent rewards
        if 'agent_rewards' in info:
            rewards = np.array([info['agent_rewards'][n] for n in self.agent_names], dtype=np.float32)
        else:
            rewards = np.full(self.n_agents, reward / self.n_agents, dtype=np.float32)
        
        # Which agents had tasks (for filtering training)
        if 'agents_with_tasks' in info:
            had_tasks = np.array([info['agents_with_tasks'][n] for n in self.agent_names], dtype=bool)
        else:
            had_tasks = np.ones(self.n_agents, dtype=bool)
        
        dones = np.full(self.n_agents, term or trunc, dtype=bool)
        return local_obs, global_obs, rewards, dones, had_tasks, info

    def step(self, actions):
        """Convenience: begin_step + complete_step combined."""
        self.begin_step()
        lo, go, r, d, had_tasks, info = self.complete_step(actions)
        # Return without had_tasks for backward compatibility
        return lo, go, r, d, info

    def get_action_masks(self):
        """Return which actions are valid. All actions always valid (queue check is stale)."""
        masks = np.ones((self.n_agents, self.n_actions), dtype=bool)
        return masks

def make_multi_drone_env(seed=42, **kwargs):
    mpd = AgentRuntime("MPD", 20.0, 20.0, 0.005, 1.0, 1, 0.0, ["drone", "any"])
    lpd = AgentRuntime("LPD", 10.0, 5.0, 0.010, 1.0, 1, 0.0, ["drone", "any"])
    gpu = AgentRuntime("GPU", 40.0, 40.0, 0.0, 1.0, 4, 0.0, ["gpu", "any"])
    cloud = AgentRuntime("Cloud", 1000.0, 100.0, 0.0, 1.0, 8, 5.0, ["cloud", "any"])
    
    sim = OffloadSim(mpd, lpd, gpu, cloud, kwargs.get("chunks_per_drone", [5,5,5,5]),
                    kwargs.get("chunk_size_mb", 10.0), kwargs.get("base_deadline", 50),
                    base_wavelength=kwargs.get("base_wavelength", 50.0),
                    base_amplitude=kwargs.get("base_amplitude", 3.0),
                    base_baseline=kwargs.get("base_baseline", 8.0),
                    network_noise_std=kwargs.get("network_noise_std", 0.3),
                    max_steps=kwargs.get("max_steps", 100),
                    streaming_enabled=kwargs.get("streaming_enabled", False),
                    stream_interval=kwargs.get("stream_interval", 5),
                    workload_band=kwargs.get("workload_band"), seed=seed)
    return MultiDroneOffloadEnv(sim)

# =============================================================================
# MARL TRAINING (requires torch)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    @dataclass
    class TrainingConfig:
        n_envs: int = 8
        total_timesteps: int = 500_000
        n_steps: int = 128
        n_epochs: int = 4
        batch_size: int = 256
        gamma: float = 0.99
        gae_lambda: float = 0.95
        clip_epsilon: float = 0.2
        vf_coef: float = 0.5
        ent_coef: float = 0.02  # Lower entropy - policies are learning correct behavior
        max_grad_norm: float = 0.5
        lr_actor: float = 3e-4  # Standard PPO learning rate (reduced from 1e-3)
        lr_critic: float = 5e-4  # Slightly higher for faster value learning
        hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
        log_interval: int = 10
        eval_interval: int = 50
        save_interval: int = 100
        max_steps_per_episode: int = 100
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden, output_gain=1.0):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                linear = nn.Linear(prev, h)
                # Orthogonal initialization for better gradient flow
                nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
                layers.extend([linear, nn.ReLU()])
                prev = h
            # Final layer with configurable gain
            final = nn.Linear(prev, out_dim)
            nn.init.orthogonal_(final.weight, gain=output_gain)
            nn.init.constant_(final.bias, 0.0)
            layers.append(final)
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class Critic(nn.Module):
        def __init__(self, obs_dim, hidden, n_agents=4):
            super().__init__()
            self.n_agents = n_agents
            # Output one value per agent for proper credit assignment
            self.net = MLP(obs_dim, n_agents, hidden, output_gain=1.0)
        def forward(self, x):
            return self.net(x)  # Shape: (batch, n_agents)

    class Actor(nn.Module):
        def __init__(self, obs_dim, n_actions, hidden):
            super().__init__()
            # Small output gain (0.01) for near-uniform initial policy
            self.net = MLP(obs_dim, n_actions, hidden, output_gain=0.01)
        def forward(self, x, mask=None):
            logits = self.net(x)
            if mask is not None:
                # Convert mask to boolean if needed (handles float tensors)
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                logits = logits.masked_fill(~mask_bool, -1e9)
            return Categorical(logits=logits)

    class Buffer:
        def __init__(self, n_steps, n_envs, n_agents, local_dim, global_dim, device):
            self.n_steps, self.n_envs, self.n_agents = n_steps, n_envs, n_agents
            self.device = device
            self.local_obs = torch.zeros((n_steps, n_envs, n_agents, local_dim), device=device)
            self.global_obs = torch.zeros((n_steps, n_envs, global_dim), device=device)
            self.actions = torch.zeros((n_steps, n_envs, n_agents), dtype=torch.long, device=device)
            # Per-agent rewards for proper credit assignment!
            self.rewards = torch.zeros((n_steps, n_envs, n_agents), device=device)
            self.dones = torch.zeros((n_steps, n_envs), dtype=torch.bool, device=device)
            self.log_probs = torch.zeros((n_steps, n_envs, n_agents), device=device)
            # Per-agent values for proper advantage computation
            self.values = torch.zeros((n_steps, n_envs, n_agents), device=device)
            self.masks = torch.ones((n_steps, n_envs, n_agents, 3), dtype=torch.bool, device=device)
            # Track which agents had tasks (for filtering training)
            self.had_tasks = torch.zeros((n_steps, n_envs, n_agents), dtype=torch.bool, device=device)
            self.ptr = 0

        def add(self, lo, go, a, r, d, lp, v, m, ht):
            self.local_obs[self.ptr], self.global_obs[self.ptr] = lo, go
            self.actions[self.ptr], self.rewards[self.ptr], self.dones[self.ptr] = a, r, d
            self.log_probs[self.ptr], self.values[self.ptr], self.masks[self.ptr] = lp, v, m
            self.had_tasks[self.ptr] = ht
            self.ptr += 1

        def compute_gae(self, last_v, gamma, lam):
            """Compute per-agent GAE advantages."""
            advantages = torch.zeros_like(self.rewards)
            last_adv = torch.zeros((self.n_envs, self.n_agents), device=self.device)
            
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_v = last_v  # Shape: (n_envs, n_agents)
                else:
                    next_v = self.values[t + 1]
                
                # Expand dones to per-agent
                mask = (~self.dones[t]).float().unsqueeze(-1)  # (n_envs, 1)
                delta = self.rewards[t] + gamma * next_v * mask - self.values[t]
                last_adv = delta + gamma * lam * last_adv * mask
                advantages[t] = last_adv
            
            returns = advantages + self.values
            return returns, advantages

        def reset(self):
            self.ptr = 0

    class Trainer:
        """MARL Trainer supporting multiple algorithms.
        
        Algorithms:
        - centralized_ppo: Centralized actor (global obs) + centralized critic
          Upper bound assuming centralized control at execution
        - mappo: Decentralized actors (local obs, type sharing) + centralized critic
          Cooperative learning with coordination awareness
        - single_agent_ppo: Decentralized actors (no sharing) + decentralized critics
          Baseline without coordination - each agent optimizes own reward independently
        """
        def __init__(self, algorithm, config, make_env_fn, log_dir="runs"):
            self.algorithm, self.config = algorithm, config
            self.make_env_fn = make_env_fn
            self.device = config.device
            
            # Use different seeds for each parallel env
            self.envs = [MAPPOWrapper(make_env_fn(seed=i*100)) for i in range(config.n_envs)]
            e = self.envs[0]
            self.n_agents, self.local_dim, self.global_dim = e.n_agents, e.local_obs_dim, e.global_obs_dim
            self.n_actions = e.n_actions
            
            # Agent type mapping: which agents share parameters
            # MPD_0=0, MPD_1=1 share one actor; LPD_0=2, LPD_1=3 share another
            self.agent_types = [0, 0, 1, 1]  # 0=MPD, 1=LPD
            self.n_agent_types = 2
            
            if algorithm == "centralized_ppo":
                # Centralized: single actor using global obs
                obs_dim = self.global_dim
                self.type_actors = nn.ModuleList([
                    Actor(obs_dim, self.n_actions, config.hidden_dims) 
                    for _ in range(self.n_agent_types)
                ]).to(self.device)
                # Centralized critic with per-agent value heads
                self.critic = Critic(self.global_dim, config.hidden_dims, n_agents=self.n_agents).to(self.device)
                self.actor_opt = torch.optim.Adam(self.type_actors.parameters(), lr=config.lr_actor)
                self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
                
            elif algorithm == "mappo":
                # MAPPO: Decentralized actors with type sharing, centralized critic
                obs_dim = self.local_dim
                self.type_actors = nn.ModuleList([
                    Actor(obs_dim, self.n_actions, config.hidden_dims) 
                    for _ in range(self.n_agent_types)
                ]).to(self.device)
                # Centralized critic with per-agent value heads
                self.critic = Critic(self.global_dim, config.hidden_dims, n_agents=self.n_agents).to(self.device)
                self.actor_opt = torch.optim.Adam(self.type_actors.parameters(), lr=config.lr_actor)
                self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
                
            elif algorithm == "single_agent_ppo":
                # Single-Agent PPO: Completely independent learning per agent
                # NO parameter sharing, NO centralized critic, individual rewards only
                obs_dim = self.local_dim
                # 4 separate actors (one per agent, no sharing)
                self.agent_actors = nn.ModuleList([
                    Actor(obs_dim, self.n_actions, config.hidden_dims) 
                    for _ in range(self.n_agents)
                ]).to(self.device)
                # 4 separate critics (each uses only local obs, outputs single value)
                self.agent_critics = nn.ModuleList([
                    Critic(obs_dim, config.hidden_dims, n_agents=1)
                    for _ in range(self.n_agents)
                ]).to(self.device)
                self.actor_opt = torch.optim.Adam(self.agent_actors.parameters(), lr=config.lr_actor)
                self.critic_opt = torch.optim.Adam(self.agent_critics.parameters(), lr=config.lr_critic)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self.buffer = Buffer(config.n_steps, config.n_envs, self.n_agents,
                                self.local_dim, self.global_dim, self.device)
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.global_step = 0
            self.ep_rewards = deque(maxlen=100)
            self.episode_count = 0  # Track episodes for seed diversity
        
        def get_actor_for_agent(self, agent_idx):
            """Get the actor network for a given agent."""
            if self.algorithm == "single_agent_ppo":
                return self.agent_actors[agent_idx]
            else:
                return self.type_actors[self.agent_types[agent_idx]]

        def collect(self):
            self.buffer.reset()
            los, gos = [], []
            for i, env in enumerate(self.envs):
                seed = (self.episode_count * self.config.n_envs + i) % 100000
                lo, go, _ = env.reset(seed=seed)
                los.append(lo)
                gos.append(go)
            lo_t = torch.tensor(np.array(los), dtype=torch.float32, device=self.device)
            go_t = torch.tensor(np.array(gos), dtype=torch.float32, device=self.device)
            
            ep_reward_accum = [0.0] * self.config.n_envs
            
            for _ in range(self.config.n_steps):
                with torch.no_grad():
                    actions = torch.zeros((self.config.n_envs, self.n_agents), dtype=torch.long, device=self.device)
                    log_probs = torch.zeros((self.config.n_envs, self.n_agents), device=self.device)
                    masks = torch.ones((self.config.n_envs, self.n_agents, self.n_actions), dtype=torch.bool, device=self.device)
                    
                    # Phase 1: Advance environment (time, networks, streaming)
                    # This tells us which agents have tasks AFTER streaming
                    agents_with_tasks_list = []
                    for env in self.envs:
                        awt = env.begin_step()  # Returns bool array of which agents have tasks
                        agents_with_tasks_list.append(awt)
                    agents_with_tasks = torch.tensor(np.array(agents_with_tasks_list), dtype=torch.bool, device=self.device)
                    
                    # Only call actor for agents that have tasks to process
                    for i in range(self.n_agents):
                        has_task = agents_with_tasks[:, i]  # (n_envs,)
                        
                        if has_task.any():
                            actor = self.get_actor_for_agent(i)
                            obs = go_t if self.algorithm == "centralized_ppo" else lo_t[:, i]
                            
                            # Get observations for envs where this agent has tasks
                            obs_with_task = obs[has_task]
                            mask_with_task = masks[has_task, i]
                            
                            dist = actor(obs_with_task, mask_with_task)
                            sampled_actions = dist.sample()
                            sampled_log_probs = dist.log_prob(sampled_actions)
                            
                            # Fill in actions and log_probs for agents with tasks
                            actions[has_task, i] = sampled_actions
                            log_probs[has_task, i] = sampled_log_probs
                        # For agents without tasks: action=0 (doesn't matter), log_prob=0
                    
                    
                    # Compute values
                    if self.algorithm == "single_agent_ppo":
                        # Each agent has its own critic using local obs
                        values = torch.zeros(self.config.n_envs, self.n_agents, device=self.device)
                        for i in range(self.n_agents):
                            values[:, i] = self.agent_critics[i](lo_t[:, i]).squeeze(-1)
                    else:
                        values = self.critic(go_t)
                
                # Phase 2: Complete step with actions
                next_los, next_gos, rews_list, dones_list, had_tasks_list = [], [], [], [], []
                for j, env in enumerate(self.envs):
                    lo, go, r, d, ht, info = env.complete_step(actions[j].cpu().numpy())
                    step_reward = r.sum()
                    ep_reward_accum[j] += step_reward
                    
                    if d[0]:
                        self.ep_rewards.append(ep_reward_accum[j])
                        ep_reward_accum[j] = 0.0
                        self.episode_count += 1
                        new_seed = (self.episode_count * self.config.n_envs + j) % 100000
                        lo, go, _ = env.reset(seed=new_seed)
                    
                    next_los.append(lo)
                    next_gos.append(go)
                    rews_list.append(r)
                    dones_list.append(d[0])
                    had_tasks_list.append(ht)
                
                rews_t = torch.tensor(np.array(rews_list), dtype=torch.float32, device=self.device)
                had_tasks_t = torch.tensor(np.array(had_tasks_list), dtype=torch.bool, device=self.device)
                
                self.buffer.add(lo_t, go_t, actions, rews_t,
                               torch.tensor(dones_list, dtype=torch.bool, device=self.device),
                               log_probs, values, masks, had_tasks_t)
                
                lo_t = torch.tensor(np.array(next_los), dtype=torch.float32, device=self.device)
                go_t = torch.tensor(np.array(next_gos), dtype=torch.float32, device=self.device)
                self.global_step += self.config.n_envs
            
            with torch.no_grad():
                if self.algorithm == "single_agent_ppo":
                    # For single-agent PPO, each agent has its own critic using local obs
                    last_v = torch.zeros(self.config.n_envs, self.n_agents, device=self.device)
                    for i in range(self.n_agents):
                        last_v[:, i] = self.agent_critics[i](lo_t[:, i]).squeeze(-1)
                else:
                    last_v = self.critic(go_t)
            return self.buffer.compute_gae(last_v, self.config.gamma, self.config.gae_lambda)

        def update(self, returns, advantages):
            # Standard advantage normalization
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)
            
            n = self.config.n_steps * self.config.n_envs
            idx = np.arange(n)
            a_losses, c_losses, ents = [], [], []
            
            for _ in range(self.config.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, n, self.config.batch_size):
                    b = idx[start:start + self.config.batch_size]
                    s_idx, e_idx = b // self.config.n_envs, b % self.config.n_envs
                    
                    b_lo = self.buffer.local_obs[s_idx, e_idx]
                    b_go = self.buffer.global_obs[s_idx, e_idx]
                    b_a = self.buffer.actions[s_idx, e_idx]
                    b_old_lp = self.buffer.log_probs[s_idx, e_idx]
                    b_m = self.buffer.masks[s_idx, e_idx]
                    b_adv = advantages[s_idx, e_idx]
                    b_ret = returns[s_idx, e_idx]
                    b_ht = self.buffer.had_tasks[s_idx, e_idx]  # (batch, n_agents)
                    
                    if torch.isnan(b_lo).any() or torch.isnan(b_go).any() or torch.isnan(b_adv).any():
                        print("WARNING: NaN detected in inputs, skipping batch")
                        continue
                    
                    # Accumulate loss only for agents that had tasks
                    total_a_loss, total_ent = 0, 0
                    n_agents_with_samples = 0
                    
                    for i in range(self.n_agents):
                        had_task = b_ht[:, i]  # (batch,) which samples had tasks
                        
                        if had_task.sum() == 0:
                            continue  # No samples for this agent had tasks
                        
                        n_agents_with_samples += 1
                        actor = self.get_actor_for_agent(i)
                        obs = b_go if self.algorithm == "centralized_ppo" else b_lo[:, i]
                        
                        # Filter to only samples where agent had tasks
                        obs_f = obs[had_task]
                        mask_f = b_m[:, i][had_task]
                        action_f = b_a[:, i][had_task]
                        old_lp_f = b_old_lp[:, i][had_task]
                        adv_f = b_adv[:, i][had_task]
                        
                        dist = actor(obs_f, mask_f)
                        new_lp = dist.log_prob(action_f)
                        ent = dist.entropy().mean()
                        
                        ratio = torch.exp(torch.clamp(new_lp - old_lp_f, -20, 20))
                        ratio = torch.clamp(ratio, 0.0, 10.0)
                        
                        # Always use clipped PPO objective
                        clipped = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                        a_loss = -torch.min(ratio * adv_f, clipped * adv_f).mean()
                        
                        total_a_loss += a_loss
                        total_ent += ent
                    
                    if n_agents_with_samples == 0:
                        continue
                    
                    total_a_loss /= n_agents_with_samples
                    total_ent /= n_agents_with_samples
                    
                    if torch.isnan(total_a_loss) or torch.isnan(total_ent):
                        print("WARNING: NaN in actor loss, skipping update")
                        continue
                    
                    self.actor_opt.zero_grad()
                    (total_a_loss - self.config.ent_coef * total_ent).backward()
                    if self.algorithm == "single_agent_ppo":
                        nn.utils.clip_grad_norm_(self.agent_actors.parameters(), self.config.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.type_actors.parameters(), self.config.max_grad_norm)
                    self.actor_opt.step()
                    
                    # Critic training
                    if self.algorithm == "single_agent_ppo":
                        # Each agent has its own critic using LOCAL observations
                        total_c_loss = 0
                        for i in range(self.n_agents):
                            v_i = self.agent_critics[i](b_lo[:, i]).squeeze(-1)
                            c_loss_i = F.mse_loss(v_i, b_ret[:, i])
                            total_c_loss += c_loss_i
                        c_loss = total_c_loss / self.n_agents
                    else:
                        # Centralized critic using global observations
                        v = self.critic(b_go)
                        c_loss = F.mse_loss(v, b_ret)
                    
                    if torch.isnan(c_loss):
                        print("WARNING: NaN in critic loss, skipping update")
                        continue
                    
                    self.critic_opt.zero_grad()
                    c_loss.backward()
                    if self.algorithm == "single_agent_ppo":
                        nn.utils.clip_grad_norm_(self.agent_critics.parameters(), self.config.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                    self.critic_opt.step()
                    
                    a_losses.append(total_a_loss.item())
                    c_losses.append(c_loss.item())
                    ents.append(total_ent.item())
            
            return np.mean(a_losses) if a_losses else 0.0, np.mean(c_losses) if c_losses else 0.0, np.mean(ents) if ents else 0.0

        def train(self):
            n_updates = self.config.total_timesteps // (self.config.n_steps * self.config.n_envs)
            history = {"reward": [], "actor_loss": [], "critic_loss": [], "entropy": []}
            
            for update in range(1, n_updates + 1):
                returns, advantages = self.collect()
                a_loss, c_loss, ent = self.update(returns, advantages)
                
                mean_r = np.mean(self.ep_rewards) if self.ep_rewards else 0
                history["reward"].append(mean_r)
                history["actor_loss"].append(a_loss)
                history["critic_loss"].append(c_loss)
                history["entropy"].append(ent)
                
                if update % self.config.log_interval == 0:
                    print(f"Update {update}/{n_updates} | Steps: {self.global_step:,} | "
                          f"Reward: {mean_r:.2f} | A_Loss: {a_loss:.4f} | C_Loss: {c_loss:.4f} | Ent: {ent:.3f}")
            return history

        def evaluate(self, n_episodes=10, base_seed=1000):
            """Evaluate policy on multiple different episodes. Returns detailed stats."""
            rewards = []
            completions = []  # Track completion rate per episode
            deadline_misses_list = []  # Track deadline misses per episode
            transfer_failures_list = []  # Track transfer failures per episode
            tasks_on_time_list = []  # Track per-task on-time completion
            tasks_late_list = []      # Track per-task late completion
            on_time_rates_list = []   # Track on-time rate per episode
            all_task_latencies = []   # All task latencies across episodes
            ep_mean_latencies = []    # Mean latency per episode
            final_battery_levels = [] # Final battery levels per episode (list of dicts)
            # Only count actions when agent actually had a task
            action_counts = {0: 0, 1: 0, 2: 0}
            per_agent_actions = {}
            total_meaningful = 0
            total_steps = 0
            
            for ep in range(n_episodes):
                env = MAPPOWrapper(self.make_env_fn(seed=base_seed + ep))
                lo, go, _ = env.reset()
                lo_t = torch.tensor(lo, dtype=torch.float32, device=self.device)
                go_t = torch.tensor(go, dtype=torch.float32, device=self.device)
                ep_r, done = 0, False
                last_info = {}
                ep_latencies = []  # Latencies for this episode
                
                while not done:
                    with torch.no_grad():
                        # Phase 1: Advance environment to get accurate task info
                        agents_with_tasks = env.begin_step()
                        masks = torch.ones((self.n_agents, self.n_actions), dtype=torch.bool, device=self.device)
                        
                        actions = []
                        for i in range(self.n_agents):
                            if agents_with_tasks[i]:
                                # Agent has tasks - call actor and count decision
                                actor = self.get_actor_for_agent(i)
                                obs = go_t.unsqueeze(0) if self.algorithm == "centralized_ppo" else lo_t[i].unsqueeze(0)
                                action = actor(obs, masks[i].unsqueeze(0)).sample().item()
                                
                                # Track meaningful decision
                                action_counts[action] += 1
                                total_meaningful += 1
                                if i not in per_agent_actions:
                                    per_agent_actions[i] = {0: 0, 1: 0, 2: 0}
                                per_agent_actions[i][action] += 1
                            else:
                                # No task - action doesn't matter
                                action = 0
                            actions.append(action)
                        total_steps += self.n_agents
                    
                    # Phase 2: Complete step
                    lo, go, r, d, _, info = env.complete_step(np.array(actions))
                    ep_r += r.sum()
                    done = d[0]
                    last_info = info
                    lo_t = torch.tensor(lo, dtype=torch.float32, device=self.device)
                    go_t = torch.tensor(go, dtype=torch.float32, device=self.device)
                    
                    # Collect task latencies from this step
                    if 'completed_task_latencies' in info:
                        ep_latencies.extend(info['completed_task_latencies'])
                
                rewards.append(ep_r)
                # Calculate completion rate from final info
                if 'processed_chunks' in last_info and 'total_chunks' in last_info:
                    completion = last_info['processed_chunks'] / max(1, last_info['total_chunks']) * 100
                else:
                    completion = 100.0  # Assume complete if not tracked
                completions.append(completion)
                deadline_misses_list.append(last_info.get('deadline_misses', 0))
                transfer_failures_list.append(last_info.get('transfer_failures', 0))
                
                # Per-task deadline tracking
                tasks_on_time = last_info.get('tasks_on_time', 0)
                tasks_late = last_info.get('tasks_late', 0)
                total_tasks = last_info.get('total_chunks', 48)  # Default to typical scenario
                tasks_on_time_list.append(tasks_on_time)
                tasks_late_list.append(tasks_late)
                # on_time_rate: % of ALL tasks completed on-time
                on_time_rate = tasks_on_time / max(total_tasks, 1) * 100
                on_time_rates_list.append(on_time_rate)
                
                # Latency tracking
                all_task_latencies.extend(ep_latencies)
                ep_mean_latencies.append(np.mean(ep_latencies) if ep_latencies else 0)
                
                # Battery tracking (final levels)
                final_battery_levels.append(last_info.get('battery_levels', {}))
            
            # Compute action percentages
            total = sum(action_counts.values())
            if total > 0:
                local_pct = action_counts[0] / total * 100
                edge_pct = action_counts[1] / total * 100
                cloud_pct = action_counts[2] / total * 100
            else:
                local_pct = edge_pct = cloud_pct = 0.0
            
            # Print action distribution (meaningful decisions only)
            if total > 0:
                print(f"    Actions (when tasks): L={local_pct:.0f}% E={edge_pct:.0f}% C={cloud_pct:.0f}%")
                print(f"    ({total} decisions with tasks / {total_steps} total)")
                agent_names = ['MPD_0', 'MPD_1', 'LPD_0', 'LPD_1']
                for i in range(4):
                    if i in per_agent_actions:
                        a_total = sum(per_agent_actions[i].values())
                        if a_total > 0:
                            pcts = [per_agent_actions[i][a]/a_total*100 for a in [0, 1, 2]]
                            print(f"      {agent_names[i]}: L={pcts[0]:.0f}% E={pcts[1]:.0f}% C={pcts[2]:.0f}% ({a_total} decisions)")
            
            # Compute battery statistics
            # Mean final battery across all drones and episodes
            all_final_batteries = []
            mpd_final_batteries = []
            lpd_final_batteries = []
            for battery_dict in final_battery_levels:
                for name, level in battery_dict.items():
                    all_final_batteries.append(level)
                    if 'MPD' in name:
                        mpd_final_batteries.append(level)
                    elif 'LPD' in name:
                        lpd_final_batteries.append(level)
            
            # Return detailed stats
            stats = {
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'rewards': rewards,  # Individual episode rewards for proper std calculation
                'completion': np.mean(completions),
                'completions': completions,  # Per-episode completions
                'deadline_misses': np.mean(deadline_misses_list),
                'deadline_misses_list': deadline_misses_list,  # Per-episode
                'transfer_failures': np.mean(transfer_failures_list),
                'transfer_failures_list': transfer_failures_list,  # Per-episode
                'on_time_rate': np.mean(on_time_rates_list),
                'on_time_rates_list': on_time_rates_list,  # Per-episode
                'tasks_on_time_list': tasks_on_time_list,  # Per-episode
                'tasks_late_list': tasks_late_list,  # Per-episode
                # Latency metrics
                'avg_latency': np.mean(all_task_latencies) if all_task_latencies else 0,
                'latency_std': np.std(all_task_latencies) if all_task_latencies else 0,
                'ep_mean_latencies': ep_mean_latencies,  # Per-episode mean latencies
                # Battery metrics
                'final_battery_mean': np.mean(all_final_batteries) if all_final_batteries else 1.0,
                'final_battery_mpd': np.mean(mpd_final_batteries) if mpd_final_batteries else 1.0,
                'final_battery_lpd': np.mean(lpd_final_batteries) if lpd_final_batteries else 1.0,
                'battery_used_pct': (1.0 - np.mean(all_final_batteries)) * 100 if all_final_batteries else 0,
                # Action distribution
                'local_pct': local_pct,
                'edge_pct': edge_pct,
                'cloud_pct': cloud_pct,
                'meaningful_decisions': total,
                'total_steps': total_steps,
            }
            return stats

    def run_comparison_experiment(total_timesteps=200_000, n_seeds=3, difficulty="streaming_hard"):
        """Run comparison experiment between MARL algorithms.
        
        Algorithms compared:
        - single_agent_ppo: Decentralized learning without coordination (baseline)
        - centralized_ppo: Centralized control upper bound
        - mappo: Cooperative learning with coordination awareness (our approach)
        """
        env_config = get_difficulty_config(difficulty)
        config = TrainingConfig(total_timesteps=total_timesteps, max_steps_per_episode=env_config.get("max_steps", 100))
        
        # Create make_env that accepts seed parameter
        def make_env_with_seed(seed=None):
            if seed is None:
                seed = np.random.randint(0, 100000)
            return make_multi_drone_env(seed=seed, **env_config)
        
        # Store detailed results per algorithm
        results = {
            "single_agent_ppo": {"rewards": [], "completion": [], "deadline_misses": [], "transfer_failures": [], "on_time_rates": [], "tasks_on_time": [], "tasks_late": [], "ep_mean_latencies": [], "battery_used": [], "local_pct": [], "edge_pct": [], "cloud_pct": []},
            "centralized_ppo": {"rewards": [], "completion": [], "deadline_misses": [], "transfer_failures": [], "on_time_rates": [], "tasks_on_time": [], "tasks_late": [], "ep_mean_latencies": [], "battery_used": [], "local_pct": [], "edge_pct": [], "cloud_pct": []},
            "mappo": {"rewards": [], "completion": [], "deadline_misses": [], "transfer_failures": [], "on_time_rates": [], "tasks_on_time": [], "tasks_late": [], "ep_mean_latencies": [], "battery_used": [], "local_pct": [], "edge_pct": [], "cloud_pct": []},
        }
        
        for alg in ["single_agent_ppo", "centralized_ppo", "mappo"]:
            print(f"\n{'='*60}\nTraining {alg.upper()}\n{'='*60}")
            
            for seed in range(n_seeds):
                print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                trainer = Trainer(alg, config, make_env_with_seed, f"runs/{difficulty}/{alg}/seed_{seed}")
                trainer.train()
                
                # Evaluate with different seeds - now returns detailed stats
                stats = trainer.evaluate(n_episodes=10, base_seed=2000 + seed * 100)
                
                # Collect all episode data for proper analysis
                results[alg]["rewards"].extend(stats['rewards'])
                results[alg]["completion"].extend(stats.get('completions', [stats['completion']]))
                results[alg]["deadline_misses"].extend(stats.get('deadline_misses_list', [stats.get('deadline_misses', 0)]))
                results[alg]["transfer_failures"].extend(stats.get('transfer_failures_list', [stats.get('transfer_failures', 0)]))
                results[alg]["on_time_rates"].extend(stats.get('on_time_rates_list', [stats.get('on_time_rate', 100.0)]))
                results[alg]["tasks_on_time"].extend(stats.get('tasks_on_time_list', []))
                results[alg]["tasks_late"].extend(stats.get('tasks_late_list', []))
                results[alg]["ep_mean_latencies"].extend(stats.get('ep_mean_latencies', []))
                results[alg]["battery_used"].append(stats.get('battery_used_pct', 0))
                results[alg]["local_pct"].append(stats['local_pct'])
                results[alg]["edge_pct"].append(stats['edge_pct'])
                results[alg]["cloud_pct"].append(stats['cloud_pct'])
                
                print(f"Seed {seed + 1} eval: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
            
            # Aggregate across all episodes from all seeds
            all_rewards = results[alg]["rewards"]
            mean = np.mean(all_rewards)
            std = np.std(all_rewards)
            completion = np.mean(results[alg]["completion"])
            local_pct = np.mean(results[alg]["local_pct"])
            edge_pct = np.mean(results[alg]["edge_pct"])
            cloud_pct = np.mean(results[alg]["cloud_pct"])
            
            print(f"\n{alg}: {mean:.2f} ± {std:.2f} (Comp={completion:.1f}% L={local_pct:.0f}% E={edge_pct:.0f}% C={cloud_pct:.0f}%)")
        
        # Return summary dict compatible with baseline format
        summary = {}
        for alg in ["single_agent_ppo", "centralized_ppo", "mappo"]:
            all_rewards = results[alg]["rewards"]
            all_latencies = results[alg]["ep_mean_latencies"]
            summary[alg] = {
                'reward_mean': np.mean(all_rewards),
                'reward_std': np.std(all_rewards),
                'rewards': all_rewards,  # Per-episode rewards for detailed analysis
                'completion': np.mean(results[alg]["completion"]),
                'completions': results[alg]["completion"],  # Per-episode completions
                'deadline_misses': np.mean(results[alg].get("deadline_misses", [0])),
                'deadline_misses_list': results[alg].get("deadline_misses", []),
                'transfer_failures': np.mean(results[alg].get("transfer_failures", [0])),
                'transfer_failures_list': results[alg].get("transfer_failures", []),
                'on_time_rate': np.mean(results[alg].get("on_time_rates", [100])),
                'on_time_rates_list': results[alg].get("on_time_rates", []),
                # Latency metrics
                'avg_latency': np.mean(all_latencies) if all_latencies else 0,
                'latency_std': np.std(all_latencies) if all_latencies else 0,
                'ep_mean_latencies': all_latencies,
                # Battery metrics
                'battery_used_pct': np.mean(results[alg].get("battery_used", [0])),
                # Action distribution
                'local_pct': np.mean(results[alg]["local_pct"]),
                'edge_pct': np.mean(results[alg]["edge_pct"]),
                'cloud_pct': np.mean(results[alg]["cloud_pct"]),
            }
        
        return summary
    
    def run_multi_scenario_experiment(
        scenarios=None,
        total_timesteps=200_000,
        n_seeds=3,
        n_eval_episodes=10,
        output_dir="multi_scenario_results"
    ):
        """Train and evaluate MARL algorithms across multiple network scenarios.
        
        This provides comprehensive analysis:
        1. Train each algorithm on each scenario
        2. Cross-evaluate: test each trained model on ALL scenarios
        3. Compute aggregate performance metrics
        4. Save detailed results for further analysis
        
        Args:
            scenarios: List of scenario names (default: key agricultural scenarios)
            total_timesteps: Training steps per scenario
            n_seeds: Number of random seeds per (algorithm, scenario) pair
            n_eval_episodes: Episodes per evaluation
            output_dir: Directory to save results
            
        Returns:
            Dict with comprehensive results
        """
        import json
        from pathlib import Path
        
        if scenarios is None:
            scenarios = ["hard", "rural_field", "variable_terrain", "network_aware"]
        
        algorithms = ["single_agent_ppo", "centralized_ppo", "mappo"]
        
        # Results structure
        results = {
            "config": {
                "scenarios": scenarios,
                "algorithms": algorithms,
                "total_timesteps": total_timesteps,
                "n_seeds": n_seeds,
                "n_eval_episodes": n_eval_episodes,
            },
            "training": {},  # {scenario: {alg: {seed: train_stats}}}
            "cross_eval": {},  # {train_scenario: {alg: {test_scenario: eval_stats}}}
            "aggregate": {},  # {alg: aggregate_stats}
        }
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("MULTI-SCENARIO MARL EXPERIMENT")
        print("=" * 70)
        print(f"Scenarios: {scenarios}")
        print(f"Algorithms: {algorithms}")
        print(f"Training steps: {total_timesteps:,}")
        print(f"Seeds: {n_seeds}")
        print()
        
        # Store trained models for cross-evaluation
        trained_models = {}  # {(scenario, alg, seed): trainer}
        
        # Phase 1: Train on each scenario
        for scenario in scenarios:
            print(f"\n{'='*70}")
            print(f"TRAINING ON: {scenario.upper()}")
            print("=" * 70)
            
            env_config = get_difficulty_config(scenario)
            config = TrainingConfig(
                total_timesteps=total_timesteps, 
                max_steps_per_episode=env_config.get("max_steps", 100)
            )
            
            results["training"][scenario] = {}
            
            def make_env_with_seed(seed=None):
                if seed is None:
                    seed = np.random.randint(0, 100000)
                return make_multi_drone_env(seed=seed, **env_config)
            
            for alg in algorithms:
                print(f"\n--- {alg} ---")
                results["training"][scenario][alg] = {}
                
                for seed in range(n_seeds):
                    print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    trainer = Trainer(
                        alg, config, make_env_with_seed, 
                        f"{output_dir}/models/{scenario}/{alg}/seed_{seed}"
                    )
                    history = trainer.train()
                    
                    # Save training curve
                    import pandas as pd
                    history_df = pd.DataFrame(history)
                    history_df.to_csv(f"{output_dir}/models/{scenario}/{alg}/seed_{seed}/training_curve.csv", index=False)
                    
                    # Store for cross-eval
                    trained_models[(scenario, alg, seed)] = trainer
                    
                    # Evaluate on training scenario
                    stats = trainer.evaluate(n_episodes=n_eval_episodes, base_seed=2000 + seed * 100)
                    results["training"][scenario][alg][seed] = {
                        "reward_mean": float(stats['reward_mean']),
                        "reward_std": float(stats['reward_std']),
                        "rewards": [float(r) for r in stats['rewards']],  # Per-episode
                        "completion": float(stats['completion']),
                        "completions": [float(c) for c in stats.get('completions', [stats['completion']])],
                        "deadline_misses": float(stats.get('deadline_misses', 0)),
                        "deadline_misses_list": [int(d) for d in stats.get('deadline_misses_list', [])],
                        "transfer_failures": float(stats.get('transfer_failures', 0)),
                        "transfer_failures_list": [int(t) for t in stats.get('transfer_failures_list', [])],
                        "local_pct": float(stats['local_pct']),
                        "edge_pct": float(stats['edge_pct']),
                        "cloud_pct": float(stats['cloud_pct']),
                    }
                    print(f"reward={stats['reward_mean']:.1f}")
        
        # Phase 2: Cross-evaluation
        print(f"\n{'='*70}")
        print("CROSS-SCENARIO EVALUATION")
        print("=" * 70)
        
        for train_scenario in scenarios:
            results["cross_eval"][train_scenario] = {}
            
            for alg in algorithms:
                results["cross_eval"][train_scenario][alg] = {}
                
                for test_scenario in scenarios:
                    # Get test environment
                    test_config = get_difficulty_config(test_scenario)
                    
                    def make_test_env(seed=None):
                        if seed is None:
                            seed = np.random.randint(0, 100000)
                        return make_multi_drone_env(seed=seed, **test_config)
                    
                    # Aggregate over seeds
                    all_rewards = []
                    all_completion = []
                    all_deadline_misses = []
                    all_transfer_failures = []
                    
                    for seed in range(n_seeds):
                        trainer = trained_models[(train_scenario, alg, seed)]
                        # Temporarily swap make_env_fn for evaluation
                        original_make_env = trainer.make_env_fn
                        trainer.make_env_fn = make_test_env
                        
                        stats = trainer.evaluate(n_episodes=n_eval_episodes, base_seed=3000 + seed * 100)
                        all_rewards.extend(stats['rewards'])
                        all_completion.extend(stats.get('completions', [stats['completion']]))
                        all_deadline_misses.extend(stats.get('deadline_misses_list', [stats.get('deadline_misses', 0)]))
                        all_transfer_failures.extend(stats.get('transfer_failures_list', [stats.get('transfer_failures', 0)]))
                        
                        trainer.make_env_fn = original_make_env
                    
                    results["cross_eval"][train_scenario][alg][test_scenario] = {
                        "reward_mean": float(np.mean(all_rewards)),
                        "reward_std": float(np.std(all_rewards)),
                        "rewards": [float(r) for r in all_rewards],  # Per-episode rewards
                        "completion": float(np.mean(all_completion)),
                        "completions": [float(c) for c in all_completion],
                        "deadline_misses": float(np.mean(all_deadline_misses)) if all_deadline_misses else 0,
                        "deadline_misses_list": [int(d) for d in all_deadline_misses],
                        "transfer_failures": float(np.mean(all_transfer_failures)) if all_transfer_failures else 0,
                        "transfer_failures_list": [int(t) for t in all_transfer_failures],
                    }
                    
                    marker = "◀" if train_scenario == test_scenario else ""
                    print(f"  Train:{train_scenario:15s} Alg:{alg:18s} Test:{test_scenario:15s} → "
                          f"reward={np.mean(all_rewards):7.1f} {marker}")
        
        # Phase 3: Compute aggregate statistics
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS")
        print("=" * 70)
        
        for alg in algorithms:
            # Aggregate across all training scenarios (generalization)
            all_cross_rewards = []
            same_scenario_rewards = []
            diff_scenario_rewards = []
            
            for train_scenario in scenarios:
                for test_scenario in scenarios:
                    r = results["cross_eval"][train_scenario][alg][test_scenario]["reward_mean"]
                    all_cross_rewards.append(r)
                    if train_scenario == test_scenario:
                        same_scenario_rewards.append(r)
                    else:
                        diff_scenario_rewards.append(r)
            
            results["aggregate"][alg] = {
                "mean_all": float(np.mean(all_cross_rewards)),
                "std_all": float(np.std(all_cross_rewards)),
                "mean_same_scenario": float(np.mean(same_scenario_rewards)),
                "mean_diff_scenario": float(np.mean(diff_scenario_rewards)),
                "generalization_gap": float(np.mean(same_scenario_rewards) - np.mean(diff_scenario_rewards)),
            }
            
            print(f"\n{alg}:")
            print(f"  Overall mean:        {results['aggregate'][alg]['mean_all']:7.1f}")
            print(f"  Same-scenario mean:  {results['aggregate'][alg]['mean_same_scenario']:7.1f}")
            print(f"  Cross-scenario mean: {results['aggregate'][alg]['mean_diff_scenario']:7.1f}")
            print(f"  Generalization gap:  {results['aggregate'][alg]['generalization_gap']:+7.1f}")
        
        # Save results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save JSON
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved {output_dir}/results.json")
        
        # Save summary CSV
        rows = []
        for train_scenario in scenarios:
            for alg in algorithms:
                for test_scenario in scenarios:
                    row = {
                        "train_scenario": train_scenario,
                        "algorithm": alg,
                        "test_scenario": test_scenario,
                        **results["cross_eval"][train_scenario][alg][test_scenario]
                    }
                    rows.append(row)
        
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/cross_eval_matrix.csv", index=False)
        print(f"✓ Saved {output_dir}/cross_eval_matrix.csv")
        
        # Save aggregate summary
        agg_rows = []
        for alg in algorithms:
            agg_rows.append({
                "algorithm": alg,
                **results["aggregate"][alg]
            })
        agg_df = pd.DataFrame(agg_rows).sort_values("mean_all", ascending=False)
        agg_df.to_csv(f"{output_dir}/aggregate_summary.csv", index=False)
        print(f"✓ Saved {output_dir}/aggregate_summary.csv")
        
        # Save ALL per-episode rewards (for detailed statistical analysis)
        episode_rows = []
        for train_scenario in scenarios:
            for alg in algorithms:
                for test_scenario in scenarios:
                    rewards = results["cross_eval"][train_scenario][alg][test_scenario].get("rewards", [])
                    for ep_idx, reward in enumerate(rewards):
                        episode_rows.append({
                            "train_scenario": train_scenario,
                            "algorithm": alg,
                            "test_scenario": test_scenario,
                            "episode": ep_idx,
                            "reward": reward,
                        })
        episode_df = pd.DataFrame(episode_rows)
        episode_df.to_csv(f"{output_dir}/all_episode_rewards.csv", index=False)
        print(f"✓ Saved {output_dir}/all_episode_rewards.csv ({len(episode_rows)} episodes)")
        
        # Print final leaderboard
        print(f"\n{'='*70}")
        print("FINAL LEADERBOARD (by overall mean reward)")
        print("=" * 70)
        print(agg_df.to_string(index=False))
        
        # Winner announcement
        best_alg = agg_df.iloc[0]['algorithm']
        best_score = agg_df.iloc[0]['mean_all']
        print(f"\n🏆 Best Algorithm: {best_alg} (mean reward: {best_score:.1f})")
        
        return results
else:
    def run_comparison_experiment(*a, **kw):
        raise ImportError("PyTorch required. Install with: pip install torch")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("FieldVision Complete - Testing...")
    
    # Show all available scenarios
    scenarios = ["easy", "medium", "hard", "streaming_hard", "stress",
                 "rural_field", "variable_terrain", "weather_stress", "peak_survey", "network_aware"]
    print("\nAvailable scenarios:")
    for d in scenarios:
        c = get_difficulty_config(d)
        print(f"  {d:18s}: chunks={sum(c['chunks_per_drone']):2d}, deadline={c['base_deadline']:2d}, "
              f"bw={c['base_baseline']:.1f}±{c['base_amplitude']:.1f}, noise={c['network_noise_std']:.1f}")
    
    env = make_multi_drone_env(seed=42, **get_difficulty_config("network_aware"))
    w = MAPPOWrapper(env)
    lo, go, _ = w.reset()
    print(f"\nShapes: local={lo.shape}, global={go.shape}")
    
    total = 0
    for _ in range(100):
        lo, go, r, d, _ = w.step(np.random.randint(0, 3, 4))
        total += r.sum()
        if d[0]: break
    print(f"Random episode: {total:.2f}")
    print("\nDone!")
