# CARLA Simulations

This directory contains CARLA scripts for simulating selected controllers in longitudinal car-following scenarios. The simulations complement the Lean 4 formalizations by showing controller behavior in CARLA.

## Datasets

The associated trajectory analysis uses two real-world datasets:

- **HighD:** Provides highway vehicle trajectories. It does not identify whether each vehicle is human-driven or automated.
- **NRC platooning experiment:** Provides a mixed-traffic interaction with a human-driven leader and an ACC-equipped follower.

These datasets support the real-world trajectory analysis. The CARLA scripts simulate vehicle interactions.

## Files

- `HOCBF-FixedDelta.py` — CARLA simulation with a fixed-\(\delta\) HOCBF.
- `HOCBF_Delta_Bounded_Rate.py` — CARLA simulation with an adaptive-\(\delta\) HOCBF.
- `TTC-BC-HighD.py` — TTC-based barrier certificate evaluation using HighD trajectories.
- `TTC-THW-HighD.py` — Combined TTC–THW evaluation using HighD trajectories.

## Setup and running

Install CARLA 0.9.15 and use its matching Python API. Open the map and configure the scenario required by the script, then run the corresponding script. Check its comments and configuration for the settings it uses.

The HOCBF validation scenario uses Town04, runs for 30 seconds, and starts both vehicles at 10 m/s with a 25 m center-to-center distance (a 20 m bumper gap for 5 m vehicles). The leader uses autopilot.
