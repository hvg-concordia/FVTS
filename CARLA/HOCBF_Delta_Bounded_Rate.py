import argparse
from dataclasses import dataclass
from typing import NamedTuple
from PlatooningSimulator import Core, PlatooningControllers
from PlatooningSimulator.utils.param_loader import load_controller_params
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
import carla
from matplotlib.lines import Line2D

# ── Variation Selection ──
RUN_VARIATION_INDEX = 2

variations = [
    [False, True, False, False, False],
    [False, True, False, True, False],
    [False, True, True, True, True],   # Adaptive delta + CBF on all followers
    [False, False, False, False, False]
]

use_cbf_flags = variations[RUN_VARIATION_INDEX]

configuration_names = {
    0: "Adaptive HOCBF on V1",
    1: "Adaptive HOCBF on V1 and V3",
    2: "Full adaptive HOCBF",
    3: "All ACC",
}

if not 0 <= RUN_VARIATION_INDEX < len(variations):
    raise ValueError("RUN_VARIATION_INDEX must be 0, 1, 2, or 3.")

# ── CBF Parameters ──
d_target = 25.0
v_set    = 25.0
# These values match Adaptive_HOCBF.lean.
alpha1   = 1.0
alpha2   = 1.0
u_min    = -8.0
u_max    =  3.0

# Adaptive Delta Parameters (with rate limiting for formal verifiability)
delta_base = 25.0
delta_max  = 40.0
window_size = 30
M_rate = 1.0          # Maximum sampled |d(delta)/dt| in m/s
delta_ddot_max = 1.0  # M2: maximum sampled |d²(delta)/dt²| in m/s³
M_vel = 10.0          # Assumed bound on |v_L - v_i| in m/s

adaptive_zero_error_gain_margin = (
    -2.0 * delta_max * delta_ddot_max
    - 2.0 * (alpha1 + alpha2) * delta_max * M_rate
    + alpha1 * alpha2 * delta_base**2
    - 2.0 * M_vel**2
)
if adaptive_zero_error_gain_margin < 0.0:
    raise ValueError(
        "The selected adaptive parameters do not satisfy the Lean sufficient "
        "zero-error gain condition."
    )

if delta_max >= d_target:
    print(
        "WARNING: d_target - delta_max is not positive. The adaptive "
        "barrier evaluates spacing-envelope preservation but does not, with "
        "these parameters alone, certify strictly positive physical spacing."
    )

leader_accel_history = []
leader_speed_history = []


@dataclass
class AdaptiveToleranceState:
    """Sampled state maintained independently for one adjacent vehicle pair."""

    delta: float = delta_base
    delta_dot: float = 0.0
    delta_ddot: float = 0.0


class HOCBFFilterResult(NamedTuple):
    command: float
    bound: float
    branch_feasible: bool
    lean_bound_compatible: bool
    hocbf_residual: float
    zero_error_satisfied: bool
    leader_input_valid: bool
    velocity_bound_valid: bool


def update_adaptive_delta(
    state,
    accel_history,
    speed_history,
    edot=0.0,
    dt=0.1,
):
    """Update one pair-specific tolerance with bounded sampled derivatives."""

    if len(accel_history) < 5:
        raw = delta_base
    else:
        accels = np.array(accel_history)
        speeds = np.array(speed_history)

        accel_var = np.var(accels)
        jerk_proxy = np.std(np.diff(accels)) if len(accels) > 5 else 0.0
        speed_var = np.var(speeds)

        instability = 1.3 * accel_var + 0.9 * jerk_proxy + 0.55 * speed_var + 0.25 * abs(edot)
        raw = delta_base + (delta_max - delta_base) * (1 - np.exp(-instability * 2.9))
        raw = min(raw, delta_max)

    safe_dt = max(float(dt), 1e-9)
    desired_rate = float(np.clip(
        (raw - state.delta) / safe_dt,
        -M_rate,
        M_rate,
    ))

    # A stopping-distance cap prevents the sampled tolerance from reaching a
    # bound with a nonzero outward rate.  This avoids an abrupt rate reset that
    # would violate the M2 second-difference bound at the next sample.
    upper_distance = max(0.0, delta_max - state.delta)
    lower_distance = max(0.0, state.delta - delta_base)
    acceleration_step = delta_ddot_max * safe_dt

    maximum_outward_rate = max(
        0.0,
        -acceleration_step
        + np.sqrt(
            acceleration_step**2
            + 2.0 * delta_ddot_max * upper_distance
        ),
    )
    minimum_outward_rate = -max(
        0.0,
        -acceleration_step
        + np.sqrt(
            acceleration_step**2
            + 2.0 * delta_ddot_max * lower_distance
        ),
    )
    desired_rate = float(
        np.clip(
            desired_rate,
            minimum_outward_rate,
            maximum_outward_rate,
        )
    )

    # Limit the sampled change in delta_dot so that the corresponding second
    # finite difference is bounded by M2 whenever the interval constraints are
    # mutually compatible.
    next_rate = state.delta_dot + np.clip(
        desired_rate - state.delta_dot,
        -delta_ddot_max * safe_dt,
        delta_ddot_max * safe_dt,
    )

    next_rate = float(np.clip(next_rate, -M_rate, M_rate))

    delta_new = float(state.delta + next_rate * safe_dt)
    if not delta_base - 1e-9 <= delta_new <= delta_max + 1e-9:
        raise RuntimeError(
            "Adaptive-tolerance update cannot satisfy its interval and "
            "second-difference bounds simultaneously."
        )
    delta_new = float(np.clip(delta_new, delta_base, delta_max))
    delta_ddot = float((next_rate - state.delta_dot) / safe_dt)

    state.delta = delta_new
    state.delta_dot = next_rate
    state.delta_ddot = delta_ddot

    derivative_bounds_satisfied = (
        abs(state.delta_dot) <= M_rate + 1e-9
        and abs(state.delta_ddot) <= delta_ddot_max + 1e-9
    )

    return (
        state.delta,
        state.delta_dot,
        state.delta_ddot,
        derivative_bounds_satisfied,
    )


def cbf_safety_filter(
    ego_speed,
    lead_speed,
    lead_accel,
    gap,
    delta,
    delta_dot,
    delta_ddot,
    alpha1=alpha1,
    alpha2=alpha2,
    u_min=u_min,
    u_max=u_max,
):
    """
    Adaptive HOCBF filter matching the pointwise Lean control expression.

    Barrier:          h(t, e) = delta(t)**2 - e**2
    Condition used:
        h_ddot + (alpha1 + alpha2) * h_dot
            + alpha1 * alpha2 * h >= 0
    """
    e = gap - d_target
    edot = lead_speed - ego_speed
    # Nominal desired acceleration (PD + cruise)
    u_des = 0.4 * e + 0.1 * edot + 0.05 * (v_set - ego_speed)
    u_des = float(np.clip(u_des, u_min, u_max))

    h_value = delta**2 - e**2
    hocbf_num = (
        delta_dot**2
        + delta * delta_ddot
        - edot**2
        + (alpha1 + alpha2) * delta * delta_dot
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
        u_bound = np.nan
        zero_residual = 2.0 * hocbf_num
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


# ── Load params ──
cfg = load_controller_params(acc_config=4)

def v_ref_cruise(predecessor, ego):
    tau = cfg["tau"]; h = cfg["h"]; c = cfg["c"]; length = cfg["vehicle_length"]
    delta_i = ego.distance_to(predecessor)
    v_ref_ms = tau / h * (predecessor.speed - ego.speed + c * (delta_i - length - h * ego.speed)) + ego.speed
    return v_ref_ms * 3.6


def signed_longitudinal_acceleration(vehicle):
    """Project CARLA acceleration onto the vehicle velocity direction."""

    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    speed = np.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
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


def acceleration_to_control(acceleration_command, steering_command):
    """Map a physically admissible acceleration command to CARLA actuation."""

    bounded_acceleration = float(
        np.clip(acceleration_command, u_min, u_max)
    )
    control = carla.VehicleControl()
    control.steer = float(np.clip(steering_command, -1.0, 1.0))

    if bounded_acceleration > 0.0:
        control.throttle = min(1.0, bounded_acceleration / u_max)
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


# ── Simulation Setup ──
simulation_length = 250
sampling_rate = 10
pid_rate = 10
N = 5

# Channels:
#   0 = gap
#   1 = speed
#   2 = realized longitudinal acceleration
#   3 = HOCBF-filtered acceleration command
#   4 = throttle
#   5 = brake
#   6 = instantaneous |v * a_long|
#   7 = pair-specific adaptive delta
#   8 = sampled delta_dot
#   9 = sampled delta_ddot
#  10 = physical branch feasibility (1 or 0)
#  11 = Lean bound-compatibility assumption (1 or 0)
#  12 = evaluated HOCBF residual
#  13 = zero-error condition (1 or 0; one away from e = 0)
#  14 = leader-acceleration validity (1 or 0)
#  15 = sampled tolerance-derivative bounds (1 or 0)
#  16 = relative-velocity bound |edot| <= M_vel (1 or 0)
number_of_samples = simulation_length * sampling_rate
log = np.full((N, 17, number_of_samples), np.nan, dtype=float)

delta_states = {
    j: AdaptiveToleranceState()
    for j in range(1, N)
}

simulation = Core.Simulation(world="Town04", dt=float(1 / sampling_rate / pid_rate),
                             large_map=False, render=True)

bps = simulation.get_vehicle_blueprints()
bp_lead = bps.filter("*vehicle.tesla.model3*")[0]
follower_bps = [bps.filter("*vehicle.audi.etron*")[0],
                bps.filter("*vehicle.bmw.*")[0],
                bps.filter("*vehicle.mini.*")[0],
                bps.filter("*vehicle.toyota.*")[0]]

spawn_points = simulation.map.get_spawn_points()
platoon = Core.Platoon(simulation)

lead_vehicle = platoon.add_lead_vehicle(bp_lead, spawn_points[30])
simulation.tick()

tm = simulation.get_trafficmanager()
tm.set_synchronous_mode(True)
tm_port = tm.get_port()
lead_vehicle.set_autopilot(True, tm_port)

followers = []
deps = [-1, 0]
pid_args_long = cfg["pid_longitudinal"]
pid_args_lat = cfg["pid_lateral"]

for i in range(N-1):
    bp_follower = follower_bps[i]
    spawn_tf = lead_vehicle.transform_ahead(-10, force_straight=True) if i == 0 else \
               followers[i-1].transform_ahead(-10, force_straight=True)
    fv = platoon.add_follower_vehicle(bp_follower, spawn_tf)
    controller = PlatooningControllers.FollowerController(
        fv, v_ref_cruise, platoon, dependencies=deps,
        pid_args_longitudinal=pid_args_long, pid_args_lateral=pid_args_lat
    )
    fv.attach_controller(controller)
    followers.append(fv)
    simulation.tick()

for fv in followers:
    fv.set_autopilot(False, tm_port)

# ── Main Loop ──
print(
    "Running Lean-aligned adaptive HOCBF configuration: "
    f"{configuration_names[RUN_VARIATION_INDEX]}"
)

sample_dt = 1.0 / sampling_rate

try:
    for step in range(simulation_length * pid_rate * sampling_rate):
        simulation.run_step(
            mode="sample" if step % pid_rate == 0 else "control"
        )

        last_vehicle = platoon[-1]
        s_transform = last_vehicle.transform_ahead(
            -14,
            force_straight=True,
        )
        s_transform.location.z += 10
        s_transform.rotation.pitch = -22
        simulation.spectator.set_transform(s_transform)

        if step % pid_rate == 0:
            log_idx = step // pid_rate

            for j, vehicle in enumerate(platoon):
                longitudinal_acceleration = (
                    signed_longitudinal_acceleration(vehicle)
                )
                log[j, 1, log_idx] = vehicle.speed
                log[j, 2, log_idx] = longitudinal_acceleration
                log[j, 6, log_idx] = abs(
                    longitudinal_acceleration * vehicle.speed
                )

                if j == 0:
                    leader_accel_history.append(
                        longitudinal_acceleration
                    )
                    leader_speed_history.append(vehicle.speed)

                    if len(leader_accel_history) > window_size:
                        leader_accel_history.pop(0)
                        leader_speed_history.pop(0)

                    # Variable human-driven-leader surrogate used consistently
                    # across controller configurations.
                    control = carla.VehicleControl(
                        steer=vehicle.get_control().steer
                    )
                    phase = (step // pid_rate) % 50
                    if phase < 12:
                        control.throttle = 0.9
                        control.brake = 0.0
                    elif phase < 22:
                        control.throttle = 0.0
                        control.brake = 0.8
                    else:
                        control.throttle = 0.45
                        control.brake = 0.0
                    vehicle.apply_control(control)
                    continue

                predecessor = platoon[j - 1]
                gap = (
                    vehicle.distance_to(predecessor)
                    - cfg["vehicle_length"]
                )
                log[j, 0, log_idx] = gap

                if not use_cbf_flags[j]:
                    continue

                current_edot = predecessor.speed - vehicle.speed
                (
                    delta_current,
                    delta_dot_current,
                    delta_ddot_current,
                    derivative_bounds_satisfied,
                ) = update_adaptive_delta(
                    state=delta_states[j],
                    accel_history=leader_accel_history,
                    speed_history=leader_speed_history,
                    edot=current_edot,
                    dt=sample_dt,
                )

                predecessor_acceleration = (
                    signed_longitudinal_acceleration(predecessor)
                )
                filter_result = cbf_safety_filter(
                    ego_speed=vehicle.speed,
                    lead_speed=predecessor.speed,
                    lead_accel=predecessor_acceleration,
                    gap=gap,
                    delta=delta_current,
                    delta_dot=delta_dot_current,
                    delta_ddot=delta_ddot_current,
                )

                log[j, 3, log_idx] = filter_result.command
                log[j, 7, log_idx] = delta_current
                log[j, 8, log_idx] = delta_dot_current
                log[j, 9, log_idx] = delta_ddot_current
                log[j, 10, log_idx] = float(
                    filter_result.branch_feasible
                )
                log[j, 11, log_idx] = float(
                    filter_result.lean_bound_compatible
                )
                log[j, 12, log_idx] = filter_result.hocbf_residual
                log[j, 13, log_idx] = float(
                    filter_result.zero_error_satisfied
                )
                log[j, 14, log_idx] = float(
                    filter_result.leader_input_valid
                )
                log[j, 15, log_idx] = float(
                    derivative_bounds_satisfied
                )
                log[j, 16, log_idx] = float(
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
                    acceleration_command=filter_result.command,
                    steering_command=steering,
                )
                vehicle.apply_control(control)

                log[j, 4, log_idx] = control.throttle
                log[j, 5, log_idx] = control.brake

        simulation.tick()

finally:
    for p in simulation.platoons:
        p.stop_all(tm_port)
    simulation.tick()
    for actor in simulation.world.get_actors():
        try:
            actor.destroy()
        except:
            pass
    simulation.release_synchronous()

    print("\n" + "=" * 75)
    print(
        "SIMULATION COMPLETE - "
        f"{configuration_names[RUN_VARIATION_INDEX]}"
    )
    print("=" * 75)

    time_vector = np.arange(number_of_samples) * sample_dt
    colors = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
    ]

    # Save every diagnostic before constructing figures.
    np.savez(
        f"adaptive_logs_variation_{RUN_VARIATION_INDEX}_lean_aligned.npz",
        time=time_vector,
        log=log,
        variation_index=RUN_VARIATION_INDEX,
        cbf_flags=np.asarray(use_cbf_flags),
        d_target=d_target,
        delta_base=delta_base,
        delta_max=delta_max,
        M_rate=M_rate,
        M2=delta_ddot_max,
        M_vel=M_vel,
        adaptive_zero_error_gain_margin=(
            adaptive_zero_error_gain_margin
        ),
        alpha1=alpha1,
        alpha2=alpha2,
        u_min=u_min,
        u_max=u_max,
    )

    fig, axs = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
    plt.subplots_adjust(hspace=0.32)

    for j in range(N):
        axs[0].plot(
            time_vector,
            log[j, 1, :] * 3.6,
            label=f"V{j}",
            color=colors[j % len(colors)],
            lw=2,
        )
    axs[0].set_ylabel("Speed (km/h)")
    axs[0].legend(loc="upper right", fontsize=9)
    axs[0].grid(alpha=0.3)

    for j in range(1, N):
        axs[1].plot(
            time_vector,
            log[j, 0, :],
            label=f"V{j} Gap",
            color=colors[j % len(colors)],
            lw=2,
        )
    axs[1].axhline(d_target, color="r", ls="--", label="d_target")
    axs[1].set_ylabel("Gap (m)")
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=0.3)

    for j in range(1, N):
        spacing_error = log[j, 0, :] - d_target
        axs[2].plot(
            time_vector,
            spacing_error,
            label=f"V{j} Error",
            color=colors[j % len(colors)],
            lw=2,
        )
        if use_cbf_flags[j]:
            axs[2].plot(
                time_vector,
                log[j, 7, :],
                color=colors[j % len(colors)],
                ls=":",
                alpha=0.65,
            )
            axs[2].plot(
                time_vector,
                -log[j, 7, :],
                color=colors[j % len(colors)],
                ls=":",
                alpha=0.65,
            )
    axs[2].axhline(0, color="k", ls=":", alpha=0.7)
    axs[2].set_ylabel("Spacing Error e(t) (m)")
    axs[2].legend(fontsize=9)
    axs[2].grid(alpha=0.3)

    for j in range(1, N):
        spacing_error = log[j, 0, :] - d_target
        active_delta = np.where(
            np.isfinite(log[j, 7, :]),
            log[j, 7, :],
            delta_base,
        )
        barrier_value = active_delta**2 - spacing_error**2
        axs[3].plot(
            time_vector,
            barrier_value,
            label=f"V{j} h(t)",
            color=colors[j % len(colors)],
            lw=2,
        )
    axs[3].axhline(0, color="r", lw=1.8)
    axs[3].set_ylabel("Barrier h(t)")
    axs[3].legend(fontsize=9)
    axs[3].grid(alpha=0.3)

    for j in range(1, N):
        if use_cbf_flags[j]:
            axs[4].plot(
                time_vector,
                log[j, 3, :],
                label=f"V{j} u_safe",
                color=colors[j % len(colors)],
                lw=2,
            )
    axs[4].axhline(0, color="k", ls="--", alpha=0.6)
    axs[4].axhline(u_min, color="r", ls=":", alpha=0.6)
    axs[4].axhline(u_max, color="r", ls=":", alpha=0.6)
    axs[4].set_ylabel("u_safe (m/s²)")
    axs[4].legend(fontsize=9)
    axs[4].grid(alpha=0.3)

    for j in range(1, N):
        if use_cbf_flags[j]:
            axs[5].plot(
                time_vector,
                log[j, 7, :],
                label=f"V{j} δ(t)",
                color=colors[j % len(colors)],
                lw=2,
            )
    axs[5].axhline(
        delta_base,
        color="k",
        ls="--",
        alpha=0.7,
        label="δ_base",
    )
    axs[5].axhline(
        delta_max,
        color="r",
        ls=":",
        alpha=0.6,
        label="δ_max",
    )
    axs[5].set_ylabel("Adaptive Spacing Tolerance δ (m)")
    axs[5].set_xlabel("Time (s)")
    axs[5].legend(loc="lower right", fontsize=9)
    axs[5].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"variation_{RUN_VARIATION_INDEX}_adaptive_hocbf_lean_aligned.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.show()

    fig_eff, ax_eff = plt.subplots(figsize=(10, 6))
    vehicle_labels = [f"V{j}" for j in range(N)]
    cumulative_efforts = [
        np.nansum(log[j, 6, :]) * sample_dt
        for j in range(N)
    ]
    bar_colors = [
        "#2ca02c" if use_cbf_flags[j] else "#1f77b4"
        for j in range(N)
    ]
    bars = ax_eff.bar(
        vehicle_labels,
        cumulative_efforts,
        color=bar_colors,
        alpha=0.85,
    )
    for bar in bars:
        height = bar.get_height()
        ax_eff.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax_eff.set_ylabel("Total Mechanical Work Demand (J/kg-equiv)")
    ax_eff.set_xlabel("Vehicle")
    ax_eff.grid(axis="y", alpha=0.6)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            lw=6,
            label="HOCBF + Adaptive δ",
        ),
        Line2D(
            [0],
            [0],
            color="#1f77b4",
            lw=6,
            label="ACC",
        ),
    ]
    ax_eff.legend(handles=legend_elements, loc="upper right")
    plt.savefig(
        f"effort_variation_{RUN_VARIATION_INDEX}_lean_aligned.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    total_work = float(np.nansum(log[:, 6, :]) * sample_dt)
    average_power = total_work / simulation_length
    max_error = float(
        np.nanmax(np.abs(log[1:, 0, :] - d_target))
    )

    active_diagnostics = [
        log[j, :, :]
        for j in range(1, N)
        if use_cbf_flags[j]
    ]
    if active_diagnostics:
        active_log = np.concatenate(active_diagnostics, axis=1)

        def finite_percentage(channel, predicate):
            values = active_log[channel, :]
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return np.nan
            return 100.0 * float(np.mean(predicate(finite_values)))

        hocbf_percent = finite_percentage(
            12,
            lambda values: values >= -1e-9,
        )
        feasibility_percent = finite_percentage(
            10,
            lambda values: values >= 0.5,
        )
        compatibility_percent = finite_percentage(
            11,
            lambda values: values >= 0.5,
        )
        derivative_percent = finite_percentage(
            15,
            lambda values: values >= 0.5,
        )
        velocity_bound_percent = finite_percentage(
            16,
            lambda values: values >= 0.5,
        )
    else:
        hocbf_percent = np.nan
        feasibility_percent = np.nan
        compatibility_percent = np.nan
        derivative_percent = np.nan
        velocity_bound_percent = np.nan

    print(f"Total Mechanical Work Demand : {total_work:.2f} J/kg-equiv")
    print(f"Max Platoon Spacing Error    : {max_error:.2f} m")
    print(f"Average Power Intensity      : {average_power:.2f} W/kg-equiv")
    if active_diagnostics:
        print(f"HOCBF Residual >= 0          : {hocbf_percent:.2f}%")
        print(f"Physical Branch Feasibility : {feasibility_percent:.2f}%")
        print(f"Lean Bound Compatibility    : {compatibility_percent:.2f}%")
        print(f"Tolerance Derivative Bounds : {derivative_percent:.2f}%")
        print(f"|edot| <= M_vel             : {velocity_bound_percent:.2f}%")
        for j in range(1, N):
            if use_cbf_flags[j]:
                print(
                    f"Final Adaptive δ for V{j}     : "
                    f"{delta_states[j].delta:.2f} m"
                )
    print("=" * 75)
