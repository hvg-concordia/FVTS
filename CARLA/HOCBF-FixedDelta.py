"""
CARLA Platooning - Lean-Aligned Fixed-Tolerance HOCBF Experiment

Parameterized HOCBF longitudinal control with the original PID lateral
steering controller.

Configurations:
    0: PHOBC on V1 only
    1: PHOBC on V1 and V3
    2: PHOBC on all followers
    3: ACC on all followers

Safety indicators:
    - Spacing envelope: |e| <= delta
    - Barrier function: h(e) = delta^2 - e^2 >= 0
    - Time to collision: TTC >= 3.0 s
    - Time headway: THW >= 1.5 s
    - HOCBF command bounds: u_min <= u_safe <= u_max
    - Pointwise HOCBF residual and actuator-feasibility diagnostics

Outputs for each individual run:
    - safety_metrics_variation_X_lean_aligned.csv
    - safety_logs_variation_X_lean_aligned.npz
    - longitudinal_safety_variation_X_lean_aligned.png

After all four configurations have been executed, the script also generates:
    1. ECDF comparison for TTC and THW.
    2. A cross-configuration safety-summary table.
    3. A grouped bar chart of TTC and THW safety satisfaction.

Run this script four times, changing RUN_VARIATION_INDEX from 0 to 3.
The comparison figures are automatically generated once all four NPZ files exist.
"""

import csv
import os
from typing import Any, NamedTuple

import carla
import matplotlib.pyplot as plt
import numpy as np

from PlatooningSimulator import Core, PlatooningControllers
from PlatooningSimulator.utils.param_loader import load_controller_params


# ============================================================
# Experiment configuration
# ============================================================

RUN_VARIATION_INDEX = 2  # Select 0, 1, 2, or 3

variations = [
    [False, True,  False, False, False],  # 0: PHOBC on V1
    [False, True,  False, True,  False],  # 1: PHOBC on V1 and V3
    [False, True,  True,  True,  True],   # 2: PHOBC on all followers
    [False, False, False, False, False],  # 3: ACC on all followers
]

configuration_names = {
    0: "PHOBC on V1",
    1: "PHOBC on V1 and V3",
    2: "Full PHOBC",
    3: "All ACC",
}

if not 0 <= RUN_VARIATION_INDEX < len(variations):
    raise ValueError(
        f"Invalid variation index {RUN_VARIATION_INDEX}. "
        "Select 0, 1, 2, or 3."
    )

use_phobc_flags = variations[RUN_VARIATION_INDEX]

print("\n" + "=" * 72)
print(f"STARTING VARIATION {RUN_VARIATION_INDEX}")
print(f"Configuration: {configuration_names[RUN_VARIATION_INDEX]}")
print("=" * 72)
print(f"PHOBC flags for V0--V4: {use_phobc_flags}")


# ============================================================
# PHOBC and safety parameters
# ============================================================

d_target = 20.0
v_set = 25.0

# Fixed tolerance used in the final fixed-envelope configuration.  Since
# delta < d_target, the associated distance interval has a positive lower end.
delta = 18.0

# These gains match HOCBF_FixedDelta.lean.
alpha1 = 1.0
alpha2 = 1.0

u_min = -8.0
u_max = 3.0
M_vel = 10.0

fixed_zero_error_gain_margin = (
    alpha1 * alpha2 * delta**2
    - 2.0 * M_vel**2
)
if fixed_zero_error_gain_margin < 0.0:
    raise ValueError(
        "The selected fixed-tolerance parameters do not satisfy the Lean "
        "sufficient zero-error gain condition."
    )

ttc_safe_threshold = 3.0
thw_safe_threshold = 1.5

closing_speed_epsilon = 0.01

# THW is evaluated only during meaningful driving conditions.
minimum_speed_for_thw = 5.0  # m/s

# Display ranges for optional time-series visualization.
ttc_plot_limit = 10.0
thw_plot_limit = 5.0

# Ranges used for the ECDF comparison.
ttc_ecdf_limit = 10.0
thw_ecdf_limit = 5.0

if delta >= d_target:
    print(
        "\nWARNING: delta is equal to or greater than d_target. "
        "The spacing envelope permits the physical gap to approach "
        "zero. The barrier h(e) = delta^2 - e^2 therefore certifies "
        "bounded spacing error, but not a strictly positive minimum gap.\n"
    )


# ============================================================
# Lean-aligned HOCBF longitudinal safety filter
# ============================================================


class HOCBFFilterResult(NamedTuple):
    """Command and pointwise assumptions associated with one filter call."""

    command: float
    bound: float
    branch_feasible: bool
    lean_bound_compatible: bool
    hocbf_residual: float
    zero_error_satisfied: bool
    leader_input_valid: bool
    velocity_bound_valid: bool


def phobc_safety_filter(
    ego_speed: float,
    lead_speed: float,
    lead_accel: float,
    gap: float,
) -> HOCBFFilterResult:
    """
    Compute the fixed-tolerance acceleration command used in the Lean model.

    Barrier:
        h(e) = delta^2 - e^2

    Recursive relative-degree-two condition:
        psi_1 = h_dot + alpha1 * h
        psi_1_dot + alpha2 * psi_1 >= 0

    Equivalently:
        h_ddot + (alpha1 + alpha2) * h_dot
            + alpha1 * alpha2 * h >= 0

    When the HOCBF half-space and the physical interval are incompatible, the
    nearest physical boundary is returned and ``branch_feasible`` is false.
    The caller logs this case instead of treating clipping as a verified step.
    """

    e = gap - d_target
    edot = lead_speed - ego_speed

    u_des = (
        0.4 * e
        + 0.1 * edot
        + 0.05 * (v_set - ego_speed)
    )

    u_des = float(np.clip(u_des, u_min, u_max))

    h_value = delta**2 - e**2
    hocbf_num = (
        -edot**2
        - (alpha1 + alpha2) * e * edot
        + 0.5 * alpha1 * alpha2 * h_value
    )

    leader_input_valid = u_min <= lead_accel <= u_max
    velocity_bound_valid = abs(edot) <= M_vel
    zero_error_satisfied = True

    if e > 0.0:
        u_bound = lead_accel - hocbf_num / e
        unbounded_command = max(u_des, u_bound)
        branch_feasible = u_bound <= u_max
        lean_bound_compatible = u_min <= u_bound <= u_max
    elif e < 0.0:
        u_bound = lead_accel - hocbf_num / e
        unbounded_command = min(u_des, u_bound)
        branch_feasible = u_bound >= u_min
        lean_bound_compatible = u_min <= u_bound <= u_max
    else:
        # At e = 0, the acceleration coefficient vanishes.  The HOCBF
        # condition is independent of the follower command.
        u_bound = np.nan
        zero_residual = (
            alpha1 * alpha2 * delta**2
            - 2.0 * edot**2
        )
        zero_error_satisfied = zero_residual >= 0.0
        branch_feasible = zero_error_satisfied
        lean_bound_compatible = zero_error_satisfied
        unbounded_command = u_des

    u_safe = float(np.clip(unbounded_command, u_min, u_max))
    hocbf_residual = float(
        2.0 * e * (u_safe - lead_accel)
        + 2.0 * hocbf_num
    )

    return HOCBFFilterResult(
        command=u_safe,
        bound=float(u_bound),
        branch_feasible=bool(branch_feasible),
        lean_bound_compatible=bool(lean_bound_compatible),
        hocbf_residual=hocbf_residual,
        zero_error_satisfied=bool(zero_error_satisfied),
        leader_input_valid=bool(leader_input_valid),
        velocity_bound_valid=bool(velocity_bound_valid),
    )


# ============================================================
# Safety-indicator functions
# ============================================================

def compute_ttc(
    gap: np.ndarray,
    ego_speed: np.ndarray,
    lead_speed: np.ndarray,
) -> np.ndarray:
    """
    Compute TTC during closing conditions:

        TTC = gap / (ego_speed - lead_speed)

    TTC is infinite when the follower is not closing on the leader.
    A non-positive gap is assigned TTC = 0.
    """

    closing_speed = ego_speed - lead_speed

    ttc = np.full_like(gap, np.inf, dtype=float)

    valid_closing_state = (
        np.isfinite(gap)
        & np.isfinite(ego_speed)
        & np.isfinite(lead_speed)
        & (gap > 0.0)
        & (closing_speed > closing_speed_epsilon)
    )

    ttc[valid_closing_state] = (
        gap[valid_closing_state]
        / closing_speed[valid_closing_state]
    )

    collision_state = np.isfinite(gap) & (gap <= 0.0)
    ttc[collision_state] = 0.0

    return ttc


def compute_time_headway(
    gap: np.ndarray,
    ego_speed: np.ndarray,
) -> np.ndarray:
    """
    Compute THW:

        THW = gap / ego_speed

    THW is evaluated only when ego speed is at least
    minimum_speed_for_thw.
    """

    thw = np.full_like(gap, np.nan, dtype=float)

    valid_state = (
        np.isfinite(gap)
        & np.isfinite(ego_speed)
        & (gap > 0.0)
        & (ego_speed >= minimum_speed_for_thw)
    )

    thw[valid_state] = gap[valid_state] / ego_speed[valid_state]

    collision_state = (
        np.isfinite(gap)
        & (gap <= 0.0)
        & np.isfinite(ego_speed)
        & (ego_speed >= minimum_speed_for_thw)
    )

    thw[collision_state] = 0.0

    return thw


def signed_longitudinal_acceleration(vehicle) -> float:
    """Project CARLA acceleration onto the vehicle velocity direction."""

    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()

    speed = np.sqrt(
        velocity.x**2
        + velocity.y**2
        + velocity.z**2
    )

    if speed <= 0.01:
        return 0.0

    return float(
        (
            acceleration.x * velocity.x
            + acceleration.y * velocity.y
            + acceleration.z * velocity.z
        )
        / speed
    )


def acceleration_to_control(
    acceleration_command: float,
    steering_command: float,
) -> carla.VehicleControl:
    """Convert a bounded acceleration command to throttle and brake."""

    bounded_acceleration = float(
        np.clip(acceleration_command, u_min, u_max)
    )

    control = carla.VehicleControl()
    control.steer = float(np.clip(steering_command, -1.0, 1.0))

    if bounded_acceleration > 0.0:
        control.throttle = min(
            1.0,
            bounded_acceleration / u_max,
        )
        control.brake = 0.0

    elif bounded_acceleration < 0.0:
        control.throttle = 0.0
        control.brake = min(
            1.0,
            abs(bounded_acceleration) / abs(u_min),
        )

    else:
        control.throttle = 0.0
        control.brake = 0.0

    return control


# ============================================================
# ACC controller parameters
# ============================================================

cfg = load_controller_params(acc_config=2)


def v_ref_cruise(predecessor, ego):
    tau = cfg["tau"]
    h = cfg["h"]
    c = cfg["c"]
    vehicle_length = cfg["vehicle_length"]

    distance = ego.distance_to(predecessor)
    predecessor_speed = predecessor.speed
    ego_speed = ego.speed

    v_ref_ms = (
        tau
        / h
        * (
            predecessor_speed
            - ego_speed
            + c
            * (
                distance
                - vehicle_length
                - h * ego_speed
            )
        )
        + ego_speed
    )

    return v_ref_ms * 3.6


# ============================================================
# Simulation parameters
# ============================================================

simulation_length = 250
sampling_rate = 10
pid_rate = 10

N = 5
number_of_samples = simulation_length * sampling_rate

# Channels:
#   0 = gap
#   1 = speed
#   2 = realized longitudinal acceleration
#   3 = bounded PHOBC command u_safe
#   4 = throttle
#   5 = brake
#   6 = active sign-branch bound
#   7 = physical branch feasibility (1 or 0)
#   8 = Lean bound-compatibility assumption (1 or 0)
#   9 = evaluated HOCBF residual
#  10 = zero-error condition (1 or 0; one away from e = 0)
#  11 = leader-acceleration validity (1 or 0)
#  12 = relative-velocity bound |edot| <= M_vel (1 or 0)
log = np.full(
    (N, 13, number_of_samples),
    np.nan,
    dtype=float,
)


# ============================================================
# CARLA setup
# ============================================================

simulation = Core.Simulation(
    world="Town04",
    dt=float(1.0 / sampling_rate / pid_rate),
    large_map=False,
    render=True,
)

blueprints = simulation.get_vehicle_blueprints()

leader_blueprint = blueprints.filter(
    "*vehicle.tesla.model3*"
)[0]

follower_blueprints = [
    blueprints.filter("*vehicle.audi.etron*")[0],
    blueprints.filter("*vehicle.bmw.*")[0],
    blueprints.filter("*vehicle.mini.*")[0],
    blueprints.filter("*vehicle.toyota.*")[0],
]

spawn_points = simulation.map.get_spawn_points()

platoon = Core.Platoon(simulation)

leader = platoon.add_lead_vehicle(
    leader_blueprint,
    spawn_points[30],
)

simulation.tick()

traffic_manager = simulation.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

tm_port = traffic_manager.get_port()

leader.set_autopilot(True, tm_port)

followers = []
dependencies = [-1, 0]

pid_args_longitudinal = cfg["pid_longitudinal"]
pid_args_lateral = cfg["pid_lateral"]

for i in range(N - 1):
    if i == 0:
        spawn_transform = leader.transform_ahead(
            -10,
            force_straight=True,
        )
    else:
        spawn_transform = followers[i - 1].transform_ahead(
            -10,
            force_straight=True,
        )

    follower = platoon.add_follower_vehicle(
        follower_blueprints[i],
        spawn_transform,
    )

    controller = PlatooningControllers.FollowerController(
        follower,
        v_ref_cruise,
        platoon,
        dependencies=dependencies,
        pid_args_longitudinal=pid_args_longitudinal,
        pid_args_lateral=pid_args_lateral,
    )

    follower.attach_controller(controller)
    followers.append(follower)

    simulation.tick()

for follower in followers:
    follower.set_autopilot(False, tm_port)


# ============================================================
# Main simulation loop
# ============================================================

print("\nRunning simulation...")

try:
    total_steps = (
        simulation_length
        * sampling_rate
        * pid_rate
    )

    for step in range(total_steps):
        simulation.run_step(
            mode="sample"
            if step % pid_rate == 0
            else "control"
        )

        last_vehicle = platoon[-1]

        spectator_transform = last_vehicle.transform_ahead(
            -14,
            force_straight=True,
        )

        spectator_transform.location.z += 10
        spectator_transform.rotation.pitch = -22

        simulation.spectator.set_transform(
            spectator_transform
        )

        if step % pid_rate == 0:
            log_idx = step // pid_rate

            for j, vehicle in enumerate(platoon):
                log[j, 1, log_idx] = vehicle.speed

                log[j, 2, log_idx] = (
                    signed_longitudinal_acceleration(vehicle)
                )

                if j == 0:
                    continue

                predecessor = platoon[j - 1]

                gap = (
                    vehicle.distance_to(predecessor)
                    - cfg["vehicle_length"]
                )

                log[j, 0, log_idx] = gap

                if use_phobc_flags[j]:
                    predecessor_acceleration = (
                        signed_longitudinal_acceleration(predecessor)
                    )

                    filter_result = phobc_safety_filter(
                        ego_speed=vehicle.speed,
                        lead_speed=predecessor.speed,
                        lead_accel=predecessor_acceleration,
                        gap=gap,
                    )

                    u_safe = filter_result.command

                    log[j, 3, log_idx] = u_safe
                    log[j, 6, log_idx] = filter_result.bound
                    log[j, 7, log_idx] = float(
                        filter_result.branch_feasible
                    )
                    log[j, 8, log_idx] = float(
                        filter_result.lean_bound_compatible
                    )
                    log[j, 9, log_idx] = (
                        filter_result.hocbf_residual
                    )
                    log[j, 10, log_idx] = float(
                        filter_result.zero_error_satisfied
                    )
                    log[j, 11, log_idx] = float(
                        filter_result.leader_input_valid
                    )
                    log[j, 12, log_idx] = float(
                        filter_result.velocity_bound_valid
                    )

                    nominal_control = (
                        vehicle.controller.run_step()
                        if hasattr(vehicle.controller, "run_step")
                        else None
                    )

                    steering = (
                        nominal_control.steer
                        if nominal_control is not None
                        else 0.0
                    )

                    control = acceleration_to_control(
                        acceleration_command=u_safe,
                        steering_command=steering,
                    )

                    vehicle.apply_control(control)

                    log[j, 4, log_idx] = control.throttle
                    log[j, 5, log_idx] = control.brake

        simulation.tick()

finally:
    for active_platoon in simulation.platoons:
        active_platoon.stop_all(tm_port)

    simulation.tick()

    for actor in simulation.world.get_actors():
        try:
            actor.destroy()
        except RuntimeError:
            pass

    simulation.release_synchronous()


# ============================================================
# Post-processing
# ============================================================

print("\n" + "=" * 72)
print(f"SIMULATION COMPLETE - VARIATION {RUN_VARIATION_INDEX}")
print("=" * 72)

dt = 1.0 / sampling_rate
time_vector = np.arange(number_of_samples) * dt

spacing_error = np.full((N, number_of_samples), np.nan)
barrier_value = np.full((N, number_of_samples), np.nan)
ttc = np.full((N, number_of_samples), np.nan)
time_headway = np.full((N, number_of_samples), np.nan)

for j in range(1, N):
    gap = log[j, 0, :]
    ego_speed = log[j, 1, :]
    lead_speed = log[j - 1, 1, :]

    spacing_error[j, :] = gap - d_target

    barrier_value[j, :] = (
        delta**2
        - spacing_error[j, :]**2
    )

    ttc[j, :] = compute_ttc(
        gap=gap,
        ego_speed=ego_speed,
        lead_speed=lead_speed,
    )

    time_headway[j, :] = compute_time_headway(
        gap=gap,
        ego_speed=ego_speed,
    )


# ============================================================
# Per-vehicle safety metrics
# ============================================================

safety_results: list[dict[str, Any]] = []

for j in range(1, N):
    gap = log[j, 0, :]
    error = spacing_error[j, :]
    barrier = barrier_value[j, :]
    vehicle_ttc = ttc[j, :]
    vehicle_thw = time_headway[j, :]
    u_safe = log[j, 3, :]
    branch_feasible = log[j, 7, :]
    lean_bound_compatible = log[j, 8, :]
    hocbf_residual = log[j, 9, :]
    zero_error_satisfied = log[j, 10, :]
    leader_input_valid = log[j, 11, :]
    velocity_bound_valid = log[j, 12, :]

    valid_gap = np.isfinite(gap)
    valid_error = np.isfinite(error)
    valid_barrier = np.isfinite(barrier)

    finite_closing_ttc = vehicle_ttc[
        np.isfinite(vehicle_ttc)
    ]

    valid_thw = vehicle_thw[
        np.isfinite(vehicle_thw)
    ]

    minimum_ttc = (
        float(np.min(finite_closing_ttc))
        if finite_closing_ttc.size > 0
        else np.inf
    )

    minimum_thw = (
        float(np.min(valid_thw))
        if valid_thw.size > 0
        else np.nan
    )

    # TTC satisfaction is calculated over closing samples only.
    ttc_satisfaction = (
        100.0
        * np.mean(
            finite_closing_ttc >= ttc_safe_threshold
        )
        if finite_closing_ttc.size > 0
        else 100.0
    )

    thw_satisfaction = (
        100.0
        * np.mean(valid_thw >= thw_safe_threshold)
        if valid_thw.size > 0
        else np.nan
    )

    envelope_satisfaction = (
        100.0
        * np.mean(
            np.abs(error[valid_error]) <= delta
        )
    )

    barrier_satisfaction = (
        100.0
        * np.mean(
            barrier[valid_barrier] >= 0.0
        )
    )

    if use_phobc_flags[j]:
        valid_commands = u_safe[np.isfinite(u_safe)]
        valid_feasibility = branch_feasible[
            np.isfinite(branch_feasible)
        ]
        valid_compatibility = lean_bound_compatible[
            np.isfinite(lean_bound_compatible)
        ]
        valid_residual = hocbf_residual[
            np.isfinite(hocbf_residual)
        ]
        valid_zero_condition = zero_error_satisfied[
            np.isfinite(zero_error_satisfied)
        ]
        valid_leader_input = leader_input_valid[
            np.isfinite(leader_input_valid)
        ]
        valid_velocity_bound = velocity_bound_valid[
            np.isfinite(velocity_bound_valid)
        ]

        if valid_commands.size > 0:
            minimum_u_safe = float(np.min(valid_commands))
            maximum_u_safe = float(np.max(valid_commands))

            command_satisfaction = (
                100.0
                * np.mean(
                    (valid_commands >= u_min)
                    & (valid_commands <= u_max)
                )
            )
            branch_feasibility_satisfaction = (
                100.0 * np.mean(valid_feasibility >= 0.5)
            )
            lean_compatibility_satisfaction = (
                100.0 * np.mean(valid_compatibility >= 0.5)
            )
            hocbf_condition_satisfaction = (
                100.0 * np.mean(valid_residual >= -1e-9)
            )
            zero_error_condition_satisfaction = (
                100.0 * np.mean(valid_zero_condition >= 0.5)
            )
            leader_input_satisfaction = (
                100.0 * np.mean(valid_leader_input >= 0.5)
            )
            velocity_bound_satisfaction = (
                100.0 * np.mean(valid_velocity_bound >= 0.5)
            )
        else:
            minimum_u_safe = np.nan
            maximum_u_safe = np.nan
            command_satisfaction = np.nan
            branch_feasibility_satisfaction = np.nan
            lean_compatibility_satisfaction = np.nan
            hocbf_condition_satisfaction = np.nan
            zero_error_condition_satisfaction = np.nan
            leader_input_satisfaction = np.nan
            velocity_bound_satisfaction = np.nan
    else:
        minimum_u_safe = np.nan
        maximum_u_safe = np.nan
        command_satisfaction = np.nan
        branch_feasibility_satisfaction = np.nan
        lean_compatibility_satisfaction = np.nan
        hocbf_condition_satisfaction = np.nan
        zero_error_condition_satisfaction = np.nan
        leader_input_satisfaction = np.nan
        velocity_bound_satisfaction = np.nan

    safety_results.append(
        {
            "variation": RUN_VARIATION_INDEX,
            "configuration": configuration_names[
                RUN_VARIATION_INDEX
            ],
            "vehicle": f"V{j}",
            "controller": (
                "PHOBC"
                if use_phobc_flags[j]
                else "ACC"
            ),
            "minimum_gap_m": float(
                np.min(gap[valid_gap])
            ),
            "maximum_absolute_spacing_error_m": float(
                np.max(np.abs(error[valid_error]))
            ),
            "minimum_barrier_value": float(
                np.min(barrier[valid_barrier])
            ),
            "spacing_envelope_satisfaction_percent": float(
                envelope_satisfaction
            ),
            "barrier_satisfaction_percent": float(
                barrier_satisfaction
            ),
            "minimum_closing_ttc_s": minimum_ttc,
            "closing_ttc_satisfaction_percent": float(
                ttc_satisfaction
            ),
            "minimum_time_headway_s": minimum_thw,
            "time_headway_satisfaction_percent": float(
                thw_satisfaction
            ),
            "minimum_u_safe_mps2": minimum_u_safe,
            "maximum_u_safe_mps2": maximum_u_safe,
            "u_safe_bound_satisfaction_percent": (
                command_satisfaction
            ),
            "physical_branch_feasibility_percent": (
                branch_feasibility_satisfaction
            ),
            "lean_bound_compatibility_percent": (
                lean_compatibility_satisfaction
            ),
            "hocbf_condition_satisfaction_percent": (
                hocbf_condition_satisfaction
            ),
            "zero_error_condition_satisfaction_percent": (
                zero_error_condition_satisfaction
            ),
            "leader_input_validity_percent": (
                leader_input_satisfaction
            ),
            "relative_velocity_bound_satisfaction_percent": (
                velocity_bound_satisfaction
            ),
        }
    )


# ============================================================
# Print per-vehicle results
# ============================================================

print("\nSAFETY ANALYSIS")
print("=" * 72)

for result in safety_results:
    print(
        f"\n{result['vehicle']} "
        f"({result['controller']})"
    )

    print(
        f"  Minimum gap: "
        f"{result['minimum_gap_m']:.2f} m"
    )

    print(
        f"  Maximum |e|: "
        f"{result['maximum_absolute_spacing_error_m']:.2f} m"
    )

    print(
        f"  Envelope satisfaction: "
        f"{result['spacing_envelope_satisfaction_percent']:.2f}%"
    )

    print(
        f"  Minimum closing TTC: "
        f"{result['minimum_closing_ttc_s']:.2f} s"
    )

    print(
        f"  Closing TTC >= {ttc_safe_threshold:.1f} s: "
        f"{result['closing_ttc_satisfaction_percent']:.2f}%"
    )

    print(
        f"  Minimum THW: "
        f"{result['minimum_time_headway_s']:.2f} s"
    )

    print(
        f"  THW >= {thw_safe_threshold:.1f} s: "
        f"{result['time_headway_satisfaction_percent']:.2f}%"
    )

    if result["controller"] == "PHOBC":
        print(
            f"  u_safe range: "
            f"[{result['minimum_u_safe_mps2']:.2f}, "
            f"{result['maximum_u_safe_mps2']:.2f}] "
            f"m/s^2"
        )

        print(
            f"  HOCBF residual >= 0: "
            f"{result['hocbf_condition_satisfaction_percent']:.2f}%"
        )

        print(
            f"  Physical branch feasibility: "
            f"{result['physical_branch_feasibility_percent']:.2f}%"
        )

        print(
            f"  Lean bound compatibility: "
            f"{result['lean_bound_compatibility_percent']:.2f}%"
        )

        print(
            f"  |edot| <= M_vel: "
            f"{result['relative_velocity_bound_satisfaction_percent']:.2f}%"
        )


# ============================================================
# Save per-vehicle metrics
# ============================================================

csv_filename = (
    f"safety_metrics_variation_"
    f"{RUN_VARIATION_INDEX}_lean_aligned.csv"
)

with open(
    csv_filename,
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=safety_results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(safety_results)

print(f"\nSafety metrics saved to: {csv_filename}")


# ============================================================
# Save numerical logs
# ============================================================

npz_filename = (
    f"safety_logs_variation_"
    f"{RUN_VARIATION_INDEX}_lean_aligned.npz"
)

np.savez(
    npz_filename,
    time=time_vector,
    log=log,
    spacing_error=spacing_error,
    barrier_value=barrier_value,
    ttc=ttc,
    time_headway=time_headway,
    variation_index=RUN_VARIATION_INDEX,
    phobc_flags=np.asarray(use_phobc_flags),
    d_target=d_target,
    delta=delta,
    alpha1=alpha1,
    alpha2=alpha2,
    u_min=u_min,
    u_max=u_max,
    M_vel=M_vel,
    fixed_zero_error_gain_margin=fixed_zero_error_gain_margin,
    ttc_safe_threshold=ttc_safe_threshold,
    thw_safe_threshold=thw_safe_threshold,
    minimum_speed_for_thw=minimum_speed_for_thw,
)

print(f"Numerical logs saved to: {npz_filename}")


# ============================================================
# Main longitudinal-safety figure
# ============================================================

vehicle_colors = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
]

fig_main, axes = plt.subplots(
    5,
    1,
    figsize=(12, 12),
    sharex=True,
)

# Speed
for j in range(N):
    axes[0].plot(
        time_vector,
        log[j, 1, :] * 3.6,
        label=f"V{j}",
        linewidth=2,
    )

axes[0].set_ylabel("Speed\n(km/h)")
axes[0].legend(loc="upper right", ncol=N)
axes[0].grid(True, alpha=0.3)

# Gap
for j in range(1, N):
    axes[1].plot(
        time_vector,
        log[j, 0, :],
        label=f"V{j}",
        linewidth=2,
    )

axes[1].axhline(
    d_target,
    linestyle="--",
    label=r"$d_{\mathrm{des}}$",
)

axes[1].axhline(
    d_target + delta,
    linestyle=":",
    label=r"$d_{\mathrm{des}}+\delta$",
)

axes[1].axhline(
    d_target - delta,
    linestyle=":",
    label=r"$d_{\mathrm{des}}-\delta$",
)

axes[1].set_ylabel("Gap\n(m)")
axes[1].legend(loc="upper right", ncol=2)
axes[1].grid(True, alpha=0.3)

# Spacing error
for j in range(1, N):
    axes[2].plot(
        time_vector,
        spacing_error[j, :],
        label=f"V{j}",
        linewidth=2,
    )

axes[2].axhline(delta, linestyle=":", label=r"$+\delta$")
axes[2].axhline(-delta, linestyle=":", label=r"$-\delta$")
axes[2].axhline(0.0, linestyle="--")

axes[2].set_ylabel("Spacing error\n(m)")
axes[2].legend(loc="upper right", ncol=2)
axes[2].grid(True, alpha=0.3)

# Barrier function
for j in range(1, N):
    axes[3].plot(
        time_vector,
        barrier_value[j, :],
        label=f"V{j}",
        linewidth=2,
    )

axes[3].axhline(
    0.0,
    linestyle="--",
    label=r"$h=0$",
)

axes[3].set_ylabel("Barrier\n$h(e)$")
axes[3].legend(loc="upper right", ncol=2)
axes[3].grid(True, alpha=0.3)

# PHOBC command
phobc_curve_plotted = False

for j in range(1, N):
    if use_phobc_flags[j]:
        axes[4].plot(
            time_vector,
            log[j, 3, :],
            label=f"V{j}",
            linewidth=2,
        )
        phobc_curve_plotted = True

axes[4].axhline(u_max, linestyle=":", label=r"$u_{\max}$")
axes[4].axhline(u_min, linestyle=":", label=r"$u_{\min}$")
axes[4].axhline(0.0, linestyle="--")

if not phobc_curve_plotted:
    axes[4].text(
        0.5,
        0.5,
        "No PHOBC-controlled followers",
        transform=axes[4].transAxes,
        ha="center",
        va="center",
    )

axes[4].set_ylabel("$u_{safe}$\n(m/s$^2$)")
axes[4].set_xlabel("Time (s)")
axes[4].legend(loc="upper right", ncol=2)
axes[4].grid(True, alpha=0.3)

plt.tight_layout()

main_figure_filename = (
    f"longitudinal_safety_variation_"
    f"{RUN_VARIATION_INDEX}_lean_aligned.png"
)

plt.savefig(
    main_figure_filename,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(
    f"Main longitudinal safety plot saved to: "
    f"{main_figure_filename}"
)


# ============================================================
# Cross-configuration analysis
# Runs only when all four NPZ files exist.
# ============================================================

expected_npz_files = [
    f"safety_logs_variation_{index}_lean_aligned.npz"
    for index in range(4)
]

all_variations_available = all(
    os.path.exists(filename)
    for filename in expected_npz_files
)

if not all_variations_available:
    missing_files = [
        filename
        for filename in expected_npz_files
        if not os.path.exists(filename)
    ]

    print(
        "\nCross-configuration figures were not generated because "
        "the following runs are still missing:"
    )

    for filename in missing_files:
        print(f"  - {filename}")

else:
    print(
        "\nAll four configurations are available. "
        "Generating ECDF, summary table, and grouped bar chart."
    )

    comparison_data = {}
    summary_rows = []

    for variation_index, filename in enumerate(
        expected_npz_files
    ):
        data = np.load(filename, allow_pickle=True)

        loaded_ttc = data["ttc"]
        loaded_thw = data["time_headway"]
        loaded_error = data["spacing_error"]
        loaded_barrier = data["barrier_value"]
        loaded_log = data["log"]

        # Pool all four followers for each configuration.
        pooled_ttc = loaded_ttc[1:, :].reshape(-1)
        pooled_thw = loaded_thw[1:, :].reshape(-1)
        pooled_error = loaded_error[1:, :].reshape(-1)
        pooled_barrier = loaded_barrier[1:, :].reshape(-1)
        pooled_gap = loaded_log[1:, 0, :].reshape(-1)

        # TTC ECDF uses only finite closing-condition TTC values.
        finite_closing_ttc = pooled_ttc[
            np.isfinite(pooled_ttc)
        ]

        # THW ECDF uses only samples evaluated above the speed threshold.
        valid_thw = pooled_thw[
            np.isfinite(pooled_thw)
        ]

        valid_error = pooled_error[
            np.isfinite(pooled_error)
        ]

        valid_barrier = pooled_barrier[
            np.isfinite(pooled_barrier)
        ]

        valid_gap = pooled_gap[
            np.isfinite(pooled_gap)
        ]

        minimum_ttc = (
            float(np.min(finite_closing_ttc))
            if finite_closing_ttc.size > 0
            else np.inf
        )

        minimum_thw = (
            float(np.min(valid_thw))
            if valid_thw.size > 0
            else np.nan
        )

        ttc_satisfaction = (
            100.0
            * np.mean(
                finite_closing_ttc >= ttc_safe_threshold
            )
            if finite_closing_ttc.size > 0
            else 100.0
        )

        thw_satisfaction = (
            100.0
            * np.mean(
                valid_thw >= thw_safe_threshold
            )
            if valid_thw.size > 0
            else np.nan
        )

        envelope_satisfaction = (
            100.0
            * np.mean(np.abs(valid_error) <= delta)
        )

        barrier_satisfaction = (
            100.0
            * np.mean(valid_barrier >= 0.0)
        )

        comparison_data[variation_index] = {
            "ttc": finite_closing_ttc,
            "thw": valid_thw,
        }

        summary_rows.append(
            {
                "variation": variation_index,
                "configuration": configuration_names[
                    variation_index
                ],
                "minimum_gap_m": float(
                    np.min(valid_gap)
                ),
                "maximum_absolute_spacing_error_m": float(
                    np.max(np.abs(valid_error))
                ),
                "spacing_envelope_satisfaction_percent": float(
                    envelope_satisfaction
                ),
                "barrier_satisfaction_percent": float(
                    barrier_satisfaction
                ),
                "minimum_closing_ttc_s": minimum_ttc,
                "closing_ttc_satisfaction_percent": float(
                    ttc_satisfaction
                ),
                "minimum_time_headway_s": minimum_thw,
                "time_headway_satisfaction_percent": float(
                    thw_satisfaction
                ),
            }
        )


    # ========================================================
    # Item 2: ECDF plots for TTC and THW
    # ========================================================

    def empirical_cdf(
        values: np.ndarray,
        upper_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute an ECDF after restricting the displayed x-range.

        Values above the upper limit are retained in the denominator,
        so the ECDF value at the upper limit still reflects the full
        sample distribution.
        """

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            return np.array([]), np.array([])

        sorted_values = np.sort(finite_values)

        display_values = sorted_values[
            sorted_values <= upper_limit
        ]

        if display_values.size == 0:
            return np.array([]), np.array([])

        cumulative_probability = (
            np.searchsorted(
                sorted_values,
                display_values,
                side="right",
            )
            / sorted_values.size
        )

        return display_values, cumulative_probability


    fig_ecdf, ecdf_axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    for variation_index in range(4):
        label = configuration_names[variation_index]

        ttc_x, ttc_y = empirical_cdf(
            comparison_data[variation_index]["ttc"],
            ttc_ecdf_limit,
        )

        if ttc_x.size > 0:
            ecdf_axes[0].step(
                ttc_x,
                ttc_y,
                where="post",
                linewidth=2,
                label=label,
            )

        thw_x, thw_y = empirical_cdf(
            comparison_data[variation_index]["thw"],
            thw_ecdf_limit,
        )

        if thw_x.size > 0:
            ecdf_axes[1].step(
                thw_x,
                thw_y,
                where="post",
                linewidth=2,
                label=label,
            )

    ecdf_axes[0].axvline(
        ttc_safe_threshold,
        linestyle="--",
        linewidth=1.5,
        label=r"TTC threshold = 3 s",
    )

    ecdf_axes[0].set_xlim(0.0, ttc_ecdf_limit)
    ecdf_axes[0].set_ylim(0.0, 1.0)
    ecdf_axes[0].set_xlabel("Time to Collision (s)")
    ecdf_axes[0].set_ylabel("Empirical CDF")
    ecdf_axes[0].grid(True, alpha=0.3)
    ecdf_axes[0].legend(loc="lower right")

    ecdf_axes[1].axvline(
        thw_safe_threshold,
        linestyle="--",
        linewidth=1.5,
        label=r"THW threshold = 1.5 s",
    )

    ecdf_axes[1].set_xlim(0.0, thw_ecdf_limit)
    ecdf_axes[1].set_ylim(0.0, 1.0)
    ecdf_axes[1].set_xlabel("Time Headway (s)")
    ecdf_axes[1].set_ylabel("Empirical CDF")
    ecdf_axes[1].grid(True, alpha=0.3)
    ecdf_axes[1].legend(loc="lower right")

    plt.tight_layout()

    ecdf_filename = "ttc_thw_ecdf_comparison_lean_aligned.png"

    plt.savefig(
        ecdf_filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(f"ECDF comparison saved to: {ecdf_filename}")


    # ========================================================
    # Item 3: Cross-configuration summary table
    # ========================================================

    summary_csv_filename = (
        "cross_configuration_safety_summary_lean_aligned.csv"
    )

    with open(
        summary_csv_filename,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"Cross-configuration summary saved to: "
        f"{summary_csv_filename}"
    )

    table_columns = [
        "Configuration",
        "Min. gap\n(m)",
        "Max. |e|\n(m)",
        "Envelope\nsafe (%)",
        "Min. TTC\n(s)",
        "TTC safe\n(%)",
        "Min. THW\n(s)",
        "THW safe\n(%)",
    ]

    table_data = []

    for row in summary_rows:
        table_data.append(
            [
                row["configuration"],
                f"{row['minimum_gap_m']:.2f}",
                (
                    f"{row['maximum_absolute_spacing_error_m']:.2f}"
                ),
                (
                    f"{row['spacing_envelope_satisfaction_percent']:.2f}"
                ),
                f"{row['minimum_closing_ttc_s']:.2f}",
                (
                    f"{row['closing_ttc_satisfaction_percent']:.2f}"
                ),
                f"{row['minimum_time_headway_s']:.2f}",
                (
                    f"{row['time_headway_satisfaction_percent']:.2f}"
                ),
            ]
        )

    fig_table, table_axis = plt.subplots(
        figsize=(13, 3.2)
    )

    table_axis.axis("off")

    safety_table = table_axis.table(
        cellText=table_data,
        colLabels=table_columns,
        cellLoc="center",
        loc="center",
    )

    safety_table.auto_set_font_size(False)
    safety_table.set_fontsize(10)
    safety_table.scale(1.0, 1.8)

    table_axis.set_title(
        "Safety Metrics Across Controller Configurations",
        pad=12,
    )

    table_figure_filename = (
        "cross_configuration_safety_table_lean_aligned.png"
    )

    plt.tight_layout()

    plt.savefig(
        table_figure_filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Safety-summary table figure saved to: "
        f"{table_figure_filename}"
    )


    # ========================================================
    # Item 4: Grouped safety-satisfaction bar chart
    # ========================================================

    configuration_labels = [
        row["configuration"]
        for row in summary_rows
    ]

    ttc_satisfaction_values = np.array(
        [
            row["closing_ttc_satisfaction_percent"]
            for row in summary_rows
        ],
        dtype=float,
    )

    thw_satisfaction_values = np.array(
        [
            row["time_headway_satisfaction_percent"]
            for row in summary_rows
        ],
        dtype=float,
    )

    x_positions = np.arange(
        len(configuration_labels)
    )

    bar_width = 0.36

    fig_bar, bar_axis = plt.subplots(
        figsize=(10, 6)
    )

    ttc_bars = bar_axis.bar(
        x_positions - bar_width / 2,
        ttc_satisfaction_values,
        bar_width,
        label=r"TTC $\geq 3$ s",
    )

    thw_bars = bar_axis.bar(
        x_positions + bar_width / 2,
        thw_satisfaction_values,
        bar_width,
        label=r"THW $\geq 1.5$ s",
    )

    bar_axis.set_ylabel(
        "Safety-Threshold Satisfaction (%)"
    )

    bar_axis.set_xticks(x_positions)
    bar_axis.set_xticklabels(
        configuration_labels,
        rotation=15,
        ha="right",
    )

    bar_axis.set_ylim(0.0, 105.0)
    bar_axis.grid(axis="y", alpha=0.3)
    bar_axis.legend(loc="lower right")

    for bar in [*ttc_bars, *thw_bars]:
        height = bar.get_height()

        if np.isfinite(height):
            bar_axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    bar_chart_filename = (
        "ttc_thw_safety_satisfaction_comparison_lean_aligned.png"
    )

    plt.savefig(
        bar_chart_filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Grouped safety-satisfaction chart saved to: "
        f"{bar_chart_filename}"
    )
