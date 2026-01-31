# FieldVision

## What problem are we solving?

Agricultrual drones generate massive amounts of high-resolution imagery that must be processed in real-time for tasks like crop counting, health monitoring, and anomaly detection. But two practical challenges block effective development:

1 **Dynamic network conditions**: Drones flying over fields experience constantly changing wireless connectivity-bandwidth fluctuates, latency spikes, and connections drop. Static offloading rules fail under these conditions.

2 **Multi-drone coordination**: When multiple drones share edge and cloud resources, independent decision-making leads to congestion, resource contention, and degraded performance. Explicit inter-drone communication is often impractical in large agricultural environments.

## What is FieldVision?
FieldVision is a **Multi-Agent Reinforcement Learning (MARL) framework** for adaptive task offloading in drone-based precision agriculture. It enables a swarm of heterogenous drones to learn cooperative offloading policies that:

* **Minimize end-to-end latency** for time-sensitive analytics
* **Reduce deadline violations** under tight operations constraints
* **Balance energy consumption** across resource-constrained drones
* **Coordinate implicitly** without requiring inter-drone communication at runtime

FieldVision uses the **Centralized Training with Decentralized Execution (CTDE)** paradigm via **Multi-Agent Proximal Policy Optimization (MAPPO)**, allowing drones to learn coordinated behavior during training while executing independently in the field.
