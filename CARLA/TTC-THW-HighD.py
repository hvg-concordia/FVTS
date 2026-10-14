#!/usr/bin/env python3
"""HighD evaluation of the combined TTC--THW control barrier formulation.

The script is aligned with the corrected Lean development:

    h = s - tau * (v_i + max(delta_v, 0))

and with the implementation constraint

    U_imp = (-delta_v + tau*a_min + kappa*h)/(2*tau),  delta_v >= 0
            (-delta_v + kappa*h)/tau,                  delta_v < 0.

When the scalar QP is feasible, its exact solution is the projection of zero:

    a_i* = max(a_min, min(0, U_imp)).

When it is infeasible, the implementation fallback is zero.  Every fallback
sample is recorded separately and is not presented as CBF-satisfying control.

The default experiment evaluates stable leader--follower pairs, matching the
pairwise mathematical and Lean formulations.  It uses two initial conditions:
safe and feasible, and short-TTC and feasible.  Safe-start selection gives
priority to windows that later contain a recorded short-TTC episode, followed
by other closing or combined-barrier events when no such episode is available;
it never uses the controlled outcome as a selection criterion.  Full
safe-start controlled trajectories provide the primary controller evaluation,
including any later infeasibility or fallback behavior.  A separate analysis
retains only the valid-domain prefix over which the controller, leader input,
and pointwise CBF assumptions hold.  Recursive propagation stops at the first
nonpositive gap.  Larger recursively controlled vehicle groups remain
available through ``--platoon-size``, but then the initial scenario condition
must hold simultaneously for every constituent pair.

The analysis uses highD's ``dhw`` field as bumper-to-bumper spacing.  According
to the highD format specification, ``dhw`` is the distance headway in metres.
This avoids direction-dependent reconstruction from bounding-box coordinates.

Example
-------
python HighD_TTC_THW_CBF_corrected.py \
    --tracks-csv "C:/path/to/25_tracks.csv" \
    --recording-meta "C:/path/to/25_recordingMeta.csv" \
    --output-dir "TTC_THW_CBF_results"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model and numerical parameters
# ---------------------------------------------------------------------------

TAU = 2.5
KAPPA = 1.0
A_MIN = -8.0
A_MAX = 3.0
DEFAULT_DT = 0.04
SPEED_MIN = 5.0
SPEED_MAX = 40.0

SAFE_START = "safe_start"
CONFLICT_START = "feasible_conflict_start"
INFEASIBLE_START = "infeasible_start"
SCENARIO_CLASSES = (SAFE_START, CONFLICT_START, INFEASIBLE_START)
SCENARIO_LABELS = {
    SAFE_START: "Safe start",
    CONFLICT_START: "Feasible conflict start",
    INFEASIBLE_START: "Infeasible start",
}

# Only a floating-point tolerance.  The mathematical closing domain remains
# delta_v > 0, rather than the 0.1 m/s filter used in the previous script.
CLOSING_EPS = 1.0e-9
NUM_TOL = 1.0e-9
CBF_TOL = 1.0e-7

REQUIRED_COLUMNS = {
    "frame",
    "id",
    "laneId",
    "precedingId",
    "xVelocity",
    "xAcceleration",
    "dhw",
}


@dataclass
class PlatoonCandidate:
    lane_id: int
    vehicle_ids: tuple[int, ...]
    frames: np.ndarray
    direction_sign: float
    scenario_class: str = "unclassified"
    selection_score: float = np.nan
    recorded_challenge_in_window: bool = False
    recorded_challenge_samples: int = 0
    recorded_ttc_conflict_in_window: bool = False
    recorded_ttc_conflict_samples: int = 0
    recorded_closing_samples: int = 0

    @property
    def direction_name(self) -> str:
        return "Positive" if self.direction_sign > 0 else "Negative"


@dataclass
class PlatoonData:
    candidate: PlatoonCandidate
    s_original: np.ndarray
    v_follower_original: np.ndarray
    v_leader_original: np.ndarray
    a_follower_original: np.ndarray
    a_leader_original: np.ndarray


@dataclass
class ControlledData:
    s: np.ndarray
    v_follower: np.ndarray
    v_leader: np.ndarray
    control: np.ndarray
    upper: np.ndarray
    feasible: np.ndarray
    barrier: np.ndarray
    exact_cbf_residual: np.ndarray
    leader_input_valid: np.ndarray
    active: np.ndarray
    speed_domain_valid: np.ndarray
    state_domain_valid: np.ndarray
    formal_scope: np.ndarray
    gap_domain_exit: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the corrected combined TTC--THW CBF on highD."
    )
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        default=Path("25_tracks.csv"),
        help="Path to XX_tracks.csv (default: 25_tracks.csv).",
    )
    parser.add_argument(
        "--recording-meta",
        type=Path,
        default=None,
        help="Optional path to XX_recordingMeta.csv; its frameRate sets dt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TTC_THW_CBF_results"),
        help="Directory for CSV, JSON, PDF, and PNG outputs.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Sampling interval used when recording metadata is not supplied.",
    )
    parser.add_argument(
        "--platoon-size",
        type=int,
        default=2,
        help=(
            "Vehicles in each recursively controlled case (default: 2, one "
            "leader--follower pair). Values above 2 require every pair to "
            "satisfy the initial scenario conditions simultaneously."
        ),
    )
    parser.add_argument(
        "--safe-cases-per-direction",
        type=int,
        default=2,
        help=(
            "Safe-start cases selected in each travel direction for the "
            "primary controller evaluation (default: 2)."
        ),
    )
    parser.add_argument(
        "--conflict-cases-per-direction",
        type=int,
        default=1,
        help=(
            "Feasible short-TTC-start cases selected in each direction "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--infeasible-cases-per-direction",
        type=int,
        default=0,
        help=(
            "Optional infeasible-start stress cases selected in each "
            "direction (default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=20.0,
        help="Duration of each selected evaluation window in seconds.",
    )
    parser.add_argument(
        "--candidate-scan-limit",
        type=int,
        default=200,
        help=(
            "Maximum stable vehicle-group candidates screened in each direction "
            "for scenario windows (default: 200)."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=50,
        help="Minimum contiguous frames for a candidate vehicle group.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help=(
            "Allow cases within the same scenario class and direction to "
            "share vehicles."
        ),
    )
    parser.add_argument(
        "--cdf-thw-max",
        type=float,
        default=None,
        help=(
            "Optional upper THW display boundary. By default, the original "
            "automatic axis scaling is used."
        ),
    )
    parser.add_argument(
        "--cdf-ttc-max",
        type=float,
        default=30.0,
        help=(
            "Upper TTC display boundary for positive-closing samples "
            "(default: 30 s). Values above the boundary are shown at its "
            "right edge and reported in the panel annotation."
        ),
    )
    parser.add_argument(
        "--publication-platoons-per-direction",
        type=int,
        default=2,
        help=(
            "Six-vehicle platoons plotted in each travel direction using "
            "the original compact publication layout (default: 2). Set to "
            "0 to disable these supplementary figures."
        ),
    )
    parser.add_argument(
        "--publication-platoon-size",
        type=int,
        default=6,
        help=(
            "Vehicles in each supplementary publication-style platoon "
            "figure (default: 6, giving five leader--follower rows)."
        ),
    )
    parser.add_argument(
        "--publication-plot-frames",
        type=int,
        default=301,
        help=(
            "Samples displayed in each supplementary publication-style "
            "platoon figure (default: 301, indexed 0 through 300)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    return parser.parse_args()


def load_dt(recording_meta: Path | None, fallback_dt: float) -> float:
    if recording_meta is None:
        if fallback_dt <= 0:
            raise ValueError("dt must be strictly positive.")
        return float(fallback_dt)

    meta = pd.read_csv(recording_meta)
    if "frameRate" not in meta.columns or meta.empty:
        raise ValueError(
            f"{recording_meta} does not contain a nonempty frameRate column."
        )
    frame_rate = float(meta.iloc[0]["frameRate"])
    if frame_rate <= 0:
        raise ValueError("The highD frameRate must be strictly positive.")
    return 1.0 / frame_rate


def load_tracks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Tracks file not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "The tracks file is missing required highD columns: "
            + ", ".join(missing)
        )

    result = df.copy()
    for column in ["frame", "id", "laneId", "precedingId"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    for column in ["xVelocity", "xAcceleration", "dhw"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(
        subset=["frame", "id", "laneId", "precedingId", "xVelocity", "dhw"]
    )
    result = result.sort_values(["frame", "laneId", "id"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid track samples remain after input validation.")
    return result


def build_track_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index the recording once for repeated vehicle--frame extraction."""
    indexed = df.set_index(["id", "frame"]).sort_index()
    if not indexed.index.is_unique:
        raise ValueError("The tracks file contains duplicate vehicle--frame rows.")
    return indexed


def discover_candidates(
    df: pd.DataFrame,
    platoon_size: int,
    min_frames: int,
) -> list[PlatoonCandidate]:
    if platoon_size < 2:
        raise ValueError("platoon-size must be at least 2.")
    if min_frames < 2:
        raise ValueError("min-frames must be at least 2.")

    # Search stable pair relations instead of iterating over every frame--lane
    # group.  A run boundary occurs when the lane, leader, or follower changes,
    # or when successive frame numbers are not contiguous.
    relations = df.loc[
        (df["precedingId"] > 0) & (df["dhw"] > 0.0),
        ["laneId", "precedingId", "id", "frame"],
    ].sort_values(["laneId", "precedingId", "id", "frame"])

    if relations.empty:
        return []

    lane = relations["laneId"].to_numpy(dtype=np.int64)
    leader = relations["precedingId"].to_numpy(dtype=np.int64)
    follower = relations["id"].to_numpy(dtype=np.int64)
    frame = relations["frame"].to_numpy(dtype=np.int64)

    new_run = np.ones(frame.size, dtype=bool)
    new_run[1:] = (
        (lane[1:] != lane[:-1])
        | (leader[1:] != leader[:-1])
        | (follower[1:] != follower[:-1])
        | (frame[1:] != frame[:-1] + 1)
    )
    run_starts = np.flatnonzero(new_run)
    run_ends = np.concatenate((run_starts[1:] - 1, [frame.size - 1]))
    run_lengths = run_ends - run_starts + 1
    valid_runs = run_lengths >= min_frames

    stable_edges: list[tuple[int, int, int, int, int]] = []
    for start_index, end_index in zip(
        run_starts[valid_runs], run_ends[valid_runs]
    ):
        stable_edges.append(
            (
                int(lane[start_index]),
                int(leader[start_index]),
                int(follower[start_index]),
                int(frame[start_index]),
                int(frame[end_index]),
            )
        )

    print(
        f"Found {len(stable_edges):,} stable leader--follower intervals "
        f"with at least {min_frames} contiguous frames."
    )

    adjacency: dict[
        tuple[int, int], list[tuple[int, int, int]]
    ] = {}
    for lane_id, leader_id, follower_id, first_frame, last_frame in stable_edges:
        adjacency.setdefault((lane_id, leader_id), []).append(
            (follower_id, first_frame, last_frame)
        )
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (-edge[2] + edge[1], edge[0]))

    # Keep the longest common interval for each vehicle sequence and lane.
    candidate_intervals: dict[
        tuple[int, tuple[int, ...]], tuple[int, int]
    ] = {}

    def extend_chain(
        lane_id: int,
        vehicle_ids: tuple[int, ...],
        first_frame: int,
        last_frame: int,
    ) -> None:
        if len(vehicle_ids) == platoon_size:
            key = (lane_id, vehicle_ids)
            previous = candidate_intervals.get(key)
            if previous is None or (last_frame - first_frame) > (
                previous[1] - previous[0]
            ):
                candidate_intervals[key] = (first_frame, last_frame)
            return

        current_leader = vehicle_ids[-1]
        for next_follower, edge_first, edge_last in adjacency.get(
            (lane_id, current_leader), []
        ):
            if next_follower in vehicle_ids:
                continue
            common_first = max(first_frame, edge_first)
            common_last = min(last_frame, edge_last)
            if common_last - common_first + 1 < min_frames:
                continue
            extend_chain(
                lane_id,
                vehicle_ids + (next_follower,),
                common_first,
                common_last,
            )

    for lane_id, leader_id, follower_id, first_frame, last_frame in stable_edges:
        extend_chain(
            lane_id,
            (leader_id, follower_id),
            first_frame,
            last_frame,
        )

    lane_velocity = df.groupby("laneId", sort=False)["xVelocity"].median()
    candidates: list[PlatoonCandidate] = []
    for (lane_id, vehicle_ids), (first_frame, last_frame) in (
        candidate_intervals.items()
    ):
        median_velocity = float(lane_velocity.loc[lane_id])
        if not np.isfinite(median_velocity) or abs(median_velocity) <= NUM_TOL:
            continue
        direction_sign = 1.0 if median_velocity > 0 else -1.0
        candidates.append(
            PlatoonCandidate(
                lane_id=lane_id,
                vehicle_ids=vehicle_ids,
                frames=np.arange(first_frame, last_frame + 1, dtype=int),
                direction_sign=direction_sign,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.frames.size,
            item.lane_id,
            item.vehicle_ids,
        )
    )
    return candidates


def select_publication_platoons(
    candidates: list[PlatoonCandidate],
    plot_frames: int,
    cases_per_direction: int,
) -> list[PlatoonCandidate]:
    """Select nonoverlapping stable groups for publication-style figures.

    These groups supply supplementary multi-pair visualizations only.  They
    are not added to the pairwise controller-evaluation population or its
    aggregate summaries.
    """
    if plot_frames < 2:
        raise ValueError("publication-plot-frames must be at least 2.")
    if cases_per_direction < 0:
        raise ValueError(
            "publication-platoons-per-direction cannot be negative."
        )

    selected: list[PlatoonCandidate] = []
    for direction_sign in (1.0, -1.0):
        used_vehicle_ids: set[int] = set()
        direction_count = 0
        for candidate in candidates:
            if candidate.direction_sign != direction_sign:
                continue
            if candidate.frames.size < plot_frames:
                continue
            if used_vehicle_ids.intersection(candidate.vehicle_ids):
                continue

            selected.append(
                PlatoonCandidate(
                    lane_id=candidate.lane_id,
                    vehicle_ids=candidate.vehicle_ids,
                    frames=candidate.frames[:plot_frames].copy(),
                    direction_sign=candidate.direction_sign,
                    scenario_class="publication_visualization",
                    selection_score=float(candidate.frames.size),
                )
            )
            used_vehicle_ids.update(candidate.vehicle_ids)
            direction_count += 1
            if direction_count >= cases_per_direction:
                break

        direction_name = "positive" if direction_sign > 0 else "negative"
        if direction_count < cases_per_direction:
            print(
                f"Publication figures: selected {direction_count}/"
                f"{cases_per_direction} nonoverlapping {direction_name}-"
                "direction platoons."
            )

    return selected


def extract_platoon(
    indexed_tracks: pd.DataFrame,
    candidate: PlatoonCandidate,
) -> PlatoonData:
    frames = candidate.frames
    ids = candidate.vehicle_ids
    sign = candidate.direction_sign
    n_pairs = len(ids) - 1
    n_frames = len(frames)

    by_vehicle: dict[int, pd.DataFrame] = {}
    for vehicle_id in ids:
        try:
            vehicle = indexed_tracks.xs(vehicle_id, level="id")
        except KeyError as error:
            raise ValueError(f"Vehicle {vehicle_id} is absent from the tracks file.") from error
        missing_frames = np.setdiff1d(frames, vehicle.index.to_numpy())
        if missing_frames.size:
            raise ValueError(
                f"Vehicle {vehicle_id} is missing selected frames "
                f"{missing_frames[:5].tolist()}."
            )
        by_vehicle[vehicle_id] = vehicle.loc[frames]

    s_original = np.empty((n_pairs, n_frames), dtype=float)
    v_follower_original = np.empty_like(s_original)
    v_leader_original = np.empty_like(s_original)
    a_follower_original = np.empty_like(s_original)
    a_leader_original = np.empty_like(s_original)

    for pair_index in range(n_pairs):
        leader_id = ids[pair_index]
        follower_id = ids[pair_index + 1]
        leader = by_vehicle[leader_id]
        follower = by_vehicle[follower_id]

        if not np.all(follower["precedingId"].to_numpy(dtype=int) == leader_id):
            raise ValueError(
                f"Pair {leader_id}->{follower_id} is not maintained over all "
                "selected frames."
            )

        # highD's dhw is the bumper-to-bumper distance headway in metres.
        s_original[pair_index] = follower["dhw"].to_numpy(dtype=float)
        v_follower_original[pair_index] = (
            sign * follower["xVelocity"].to_numpy(dtype=float)
        )
        v_leader_original[pair_index] = (
            sign * leader["xVelocity"].to_numpy(dtype=float)
        )
        a_follower_original[pair_index] = (
            sign * follower["xAcceleration"].to_numpy(dtype=float)
        )
        a_leader_original[pair_index] = (
            sign * leader["xAcceleration"].to_numpy(dtype=float)
        )

    return PlatoonData(
        candidate=candidate,
        s_original=s_original,
        v_follower_original=v_follower_original,
        v_leader_original=v_leader_original,
        a_follower_original=a_follower_original,
        a_leader_original=a_leader_original,
    )


def barrier_value(s: np.ndarray | float, v_i: np.ndarray | float,
                  v_l: np.ndarray | float) -> np.ndarray | float:
    delta_v = np.asarray(v_i) - np.asarray(v_l)
    return np.asarray(s) - TAU * (np.asarray(v_i) + np.maximum(delta_v, 0.0))


def controller_upper(s: float, v_i: float, v_l: float) -> tuple[float, float]:
    delta_v = v_i - v_l
    h = float(barrier_value(s, v_i, v_l))
    if delta_v >= 0.0:
        upper = (-delta_v + TAU * A_MIN + KAPPA * h) / (2.0 * TAU)
    else:
        upper = (-delta_v + KAPPA * h) / TAU
    return float(upper), h


def controller_upper_array(
    s: np.ndarray,
    v_i: np.ndarray,
    v_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized implementation upper bound used during case screening."""
    delta_v = np.asarray(v_i) - np.asarray(v_l)
    h = np.asarray(barrier_value(s, v_i, v_l), dtype=float)
    upper = np.where(
        delta_v >= 0.0,
        (-delta_v + TAU * A_MIN + KAPPA * h) / (2.0 * TAU),
        (-delta_v + KAPPA * h) / TAU,
    )
    return upper, h


def select_scenario_candidates(
    indexed_tracks: pd.DataFrame,
    candidates: list[PlatoonCandidate],
    window_frames: int,
    cases_per_class_direction: dict[str, int],
    scan_limit: int,
    allow_overlap: bool,
) -> list[PlatoonCandidate]:
    """Select fixed-duration windows from explicit initial-state classes.

    Every selected start also requires the recorded first leader acceleration
    to lie within the input bounds.  A safe-start window requires every pair
    to be in the state domain with ``h >= 0`` and a feasible implemented upper
    bound.  A feasible conflict-start window requires every pair to be
    feasible and at least one short-TTC interaction at the first sample.  An
    infeasible-start window requires at least one implemented upper bound
    below ``a_min``.  Among safe starts, windows containing a later recorded
    combined-barrier violation are prioritized.  This uses only the recorded
    baseline and does not select cases according to the controlled response.
    """
    if window_frames < 2:
        raise ValueError("The analysis window must contain at least two frames.")
    for scenario_class in SCENARIO_CLASSES:
        count = cases_per_class_direction[scenario_class]
        if count < 0:
            raise ValueError("Scenario case counts cannot be negative.")
    if scan_limit < 1:
        raise ValueError("candidate-scan-limit must be at least 1.")

    proposals: dict[tuple[float, str], list[PlatoonCandidate]] = {
        (direction, scenario): []
        for direction in (1.0, -1.0)
        for scenario in SCENARIO_CLASSES
    }

    for direction_sign in (1.0, -1.0):
        direction_candidates = [
            candidate
            for candidate in candidates
            if candidate.direction_sign == direction_sign
            and candidate.frames.size >= window_frames
        ][:scan_limit]

        direction_label = "positive" if direction_sign > 0 else "negative"
        for candidate_index, candidate in enumerate(
            direction_candidates, start=1
        ):
            if candidate_index == 1 or candidate_index % 25 == 0:
                print(
                    f"  Screening {direction_label}-direction candidate "
                    f"{candidate_index}/{len(direction_candidates)}..."
                )
            platoon = extract_platoon(indexed_tracks, candidate)
            upper, h = controller_upper_array(
                platoon.s_original,
                platoon.v_follower_original,
                platoon.v_leader_original,
            )
            delta_v = (
                platoon.v_follower_original - platoon.v_leader_original
            )
            ttc_screen = np.full_like(delta_v, np.inf, dtype=float)
            np.divide(
                platoon.s_original,
                delta_v,
                out=ttc_screen,
                where=delta_v > CLOSING_EPS,
            )
            state_valid = (
                (platoon.s_original > 0.0)
                & (platoon.v_leader_original > SPEED_MIN)
                & (platoon.v_leader_original < SPEED_MAX)
                & (platoon.v_follower_original > SPEED_MIN)
                & (platoon.v_follower_original < SPEED_MAX)
            )
            all_state_valid = np.all(state_valid, axis=0)
            first_leader_input_valid = (
                (platoon.a_leader_original[0] >= A_MIN - NUM_TOL)
                & (platoon.a_leader_original[0] <= A_MAX + NUM_TOL)
            )
            feasible = upper >= A_MIN - NUM_TOL
            all_feasible = np.all(feasible, axis=0)
            all_safe = np.all(h >= -NUM_TOL, axis=0)
            short_ttc = ttc_screen < TAU
            any_short_ttc = np.any(short_ttc, axis=0)
            any_infeasible = np.any(~feasible, axis=0)
            recorded_challenge = np.any(h < -NUM_TOL, axis=0)
            recorded_closing = np.any(delta_v > CLOSING_EPS, axis=0)

            def event_window_data(
                event: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                prefix = np.concatenate(
                    ([0], np.cumsum(event, dtype=int))
                )
                next_index = np.full(event.size, event.size, dtype=int)
                next_seen = event.size
                for event_index in range(event.size - 1, -1, -1):
                    if event[event_index]:
                        next_seen = event_index
                    next_index[event_index] = next_seen
                return prefix, next_index

            challenge_prefix, next_challenge_index = event_window_data(
                recorded_challenge
            )
            short_ttc_prefix, next_short_ttc_index = event_window_data(
                any_short_ttc
            )
            closing_prefix, next_closing_index = event_window_data(
                recorded_closing
            )

            last_start = candidate.frames.size - window_frames
            valid_start = np.arange(candidate.frames.size) <= last_start
            class_masks = {
                SAFE_START: (
                    valid_start
                    & all_state_valid
                    & first_leader_input_valid
                    & all_feasible
                    & all_safe
                ),
                CONFLICT_START: (
                    valid_start
                    & all_state_valid
                    & first_leader_input_valid
                    & all_feasible
                    & any_short_ttc
                ),
                INFEASIBLE_START: (
                    valid_start
                    & all_state_valid
                    & first_leader_input_valid
                    & any_infeasible
                ),
            }

            for scenario_class, eligible in class_masks.items():
                if cases_per_class_direction[scenario_class] == 0:
                    continue
                indices = np.flatnonzero(eligible)
                if indices.size == 0:
                    continue

                recorded_challenge_samples = 0
                if scenario_class == SAFE_START:
                    window_challenge_counts = (
                        challenge_prefix[indices + window_frames]
                        - challenge_prefix[indices]
                    )
                    window_short_ttc_counts = (
                        short_ttc_prefix[indices + window_frames]
                        - short_ttc_prefix[indices]
                    )
                    window_closing_counts = (
                        closing_prefix[indices + window_frames]
                        - closing_prefix[indices]
                    )

                    # TTC-conflict windows are the most informative for the
                    # combined evaluation.  If none exist, retain a closing
                    # window with finite TTC; only then fall back to a general
                    # combined-barrier challenge or a benign safe window.
                    if np.any(window_short_ttc_counts > 0):
                        event_counts = window_short_ttc_counts
                        first_event_offsets = (
                            next_short_ttc_index[indices] - indices
                        )
                    elif np.any(window_closing_counts > 0):
                        event_counts = window_closing_counts
                        first_event_offsets = (
                            next_closing_index[indices] - indices
                        )
                    else:
                        event_counts = window_challenge_counts
                        first_event_offsets = (
                            next_challenge_index[indices] - indices
                        )

                    informative = (
                        (event_counts > 0)
                        & (first_event_offsets < window_frames)
                    )
                    if np.any(informative):
                        informative_positions = np.flatnonzero(informative)
                        # Place the first recorded event near one third of the
                        # window where possible, leaving time before and after.
                        target_offset = max(1, window_frames // 3)
                        best_position = min(
                            informative_positions,
                            key=lambda position: (
                                abs(
                                    int(first_event_offsets[position])
                                    - target_offset
                                ),
                                -int(event_counts[position]),
                                int(indices[position]),
                            ),
                        )
                        start_index = int(indices[best_position])
                    else:
                        start_index = int(indices[0])
                    recorded_challenge_samples = int(
                        challenge_prefix[start_index + window_frames]
                        - challenge_prefix[start_index]
                    )
                    recorded_ttc_conflict_samples = int(
                        short_ttc_prefix[start_index + window_frames]
                        - short_ttc_prefix[start_index]
                    )
                    recorded_closing_samples = int(
                        closing_prefix[start_index + window_frames]
                        - closing_prefix[start_index]
                    )
                    selection_score = -float(
                        1_000_000 * recorded_ttc_conflict_samples
                        + 1_000 * recorded_closing_samples
                        + recorded_challenge_samples
                    )
                else:
                    # Conflict and infeasible cases begin at the first onset
                    # of their respective class.  Controlled outcomes are not
                    # used during selection.
                    onset = eligible & ~np.concatenate(
                        ([False], eligible[:-1])
                    )
                    onset_indices = np.flatnonzero(onset)
                    start_index = int(
                        onset_indices[0] if onset_indices.size else indices[0]
                    )
                    if scenario_class == CONFLICT_START:
                        selection_score = float(
                            np.min(ttc_screen[:, start_index])
                        )
                    else:
                        selection_score = float(
                            np.min(upper[:, start_index] - A_MIN)
                        )
                recorded_challenge_samples = int(
                    challenge_prefix[start_index + window_frames]
                    - challenge_prefix[start_index]
                )
                recorded_ttc_conflict_samples = int(
                    short_ttc_prefix[start_index + window_frames]
                    - short_ttc_prefix[start_index]
                )
                recorded_closing_samples = int(
                    closing_prefix[start_index + window_frames]
                    - closing_prefix[start_index]
                )
                selected_frames = candidate.frames[
                    start_index:start_index + window_frames
                ].copy()
                proposals[(direction_sign, scenario_class)].append(
                    PlatoonCandidate(
                        lane_id=candidate.lane_id,
                        vehicle_ids=candidate.vehicle_ids,
                        frames=selected_frames,
                        direction_sign=candidate.direction_sign,
                        scenario_class=scenario_class,
                        selection_score=selection_score,
                        recorded_challenge_in_window=(
                            recorded_challenge_samples > 0
                        ),
                        recorded_challenge_samples=(
                            recorded_challenge_samples
                        ),
                        recorded_ttc_conflict_in_window=(
                            recorded_ttc_conflict_samples > 0
                        ),
                        recorded_ttc_conflict_samples=(
                            recorded_ttc_conflict_samples
                        ),
                        recorded_closing_samples=recorded_closing_samples,
                    )
                )

    selected: list[PlatoonCandidate] = []
    for direction_sign in (1.0, -1.0):
        for scenario_class in SCENARIO_CLASSES:
            target_count = cases_per_class_direction[scenario_class]
            if target_count == 0:
                continue
            ranked = sorted(
                proposals[(direction_sign, scenario_class)],
                key=lambda candidate: (
                    (
                        not candidate.recorded_ttc_conflict_in_window
                        if scenario_class == SAFE_START
                        else False
                    ),
                    (
                        -candidate.recorded_ttc_conflict_samples
                        if scenario_class == SAFE_START
                        else 0
                    ),
                    (
                        candidate.recorded_closing_samples == 0
                        if scenario_class == SAFE_START
                        else False
                    ),
                    (
                        -candidate.recorded_closing_samples
                        if scenario_class == SAFE_START
                        else 0
                    ),
                    (
                        not candidate.recorded_challenge_in_window
                        if scenario_class == SAFE_START
                        else False
                    ),
                    (
                        -candidate.recorded_challenge_samples
                        if scenario_class == SAFE_START
                        else 0
                    ),
                    int(candidate.frames[0]),
                    candidate.lane_id,
                    candidate.vehicle_ids,
                ),
            )
            used_ids: set[int] = set()
            chosen: list[PlatoonCandidate] = []
            for candidate in ranked:
                candidate_ids = set(candidate.vehicle_ids)
                if not allow_overlap and candidate_ids.intersection(used_ids):
                    continue
                chosen.append(candidate)
                used_ids.update(candidate_ids)
                if len(chosen) == target_count:
                    break
            selected.extend(chosen)
            print(
                f"Selected {len(chosen)} {SCENARIO_LABELS[scenario_class].lower()} "
                f"case(s) in the "
                f"{'positive' if direction_sign > 0 else 'negative'} direction."
            )
    return selected


def selected_control(s: float, v_i: float, v_l: float) -> tuple[float, float, bool, float]:
    upper, h = controller_upper(s, v_i, v_l)
    feasible = A_MIN <= upper + NUM_TOL
    if feasible:
        control = max(A_MIN, min(0.0, upper))
    else:
        # This is the implementation fallback, not a CBF-certified control.
        control = 0.0
    return float(control), upper, bool(feasible), h


def exact_barrier_rate(delta_v: float, a_i: float, a_l: float) -> float:
    if delta_v > 0.0:
        return -delta_v - TAU * a_i - TAU * (a_i - a_l)
    if delta_v < 0.0:
        return -delta_v - TAU * a_i
    return -TAU * a_i - TAU * max(a_i - a_l, 0.0)


def simulate_controlled(platoon: PlatoonData, dt: float) -> ControlledData:
    n_pairs, n_frames = platoon.s_original.shape
    s = np.full_like(platoon.s_original, np.nan)
    v_follower = np.full_like(platoon.v_follower_original, np.nan)
    v_leader = np.full_like(platoon.v_leader_original, np.nan)
    control = np.full_like(s, np.nan)
    upper = np.full_like(s, np.nan)
    feasible = np.zeros_like(s, dtype=bool)
    barrier = np.full_like(s, np.nan)
    exact_cbf_residual = np.full_like(s, np.nan)
    leader_input_valid = np.zeros_like(s, dtype=bool)
    active = np.zeros_like(s, dtype=bool)
    speed_domain_valid = np.zeros_like(s, dtype=bool)
    state_domain_valid = np.zeros_like(s, dtype=bool)
    formal_scope = np.zeros_like(s, dtype=bool)
    gap_domain_exit = np.zeros_like(s, dtype=bool)

    s[:, 0] = platoon.s_original[:, 0]
    v_follower[:, 0] = platoon.v_follower_original[:, 0]
    formal_prefix_open = True

    for time_index in range(n_frames):
        # The first leader follows its recorded trajectory.  Every subsequent
        # leader is the recursively controlled predecessor.
        v_leader[0, time_index] = platoon.v_leader_original[0, time_index]
        if n_pairs > 1:
            v_leader[1:, time_index] = v_follower[:-1, time_index]

        current_gap_valid = s[:, time_index] > 0.0
        current_speed_valid = (
            (v_leader[:, time_index] > SPEED_MIN)
            & (v_leader[:, time_index] < SPEED_MAX)
            & (v_follower[:, time_index] > SPEED_MIN)
            & (v_follower[:, time_index] < SPEED_MAX)
        )
        speed_domain_valid[:, time_index] = current_speed_valid
        state_domain_valid[:, time_index] = (
            current_gap_valid & current_speed_valid
        )

        # Once any pair reaches a nonpositive spacing, the recursively
        # propagated platoon no longer has a physically interpretable state.
        # Preserve the first domain-exit sample and stop the propagation.
        if not np.all(current_gap_valid):
            gap_domain_exit[:, time_index] = ~current_gap_valid
            barrier[:, time_index] = barrier_value(
                s[:, time_index],
                v_follower[:, time_index],
                v_leader[:, time_index],
            )
            break

        active[:, time_index] = True

        for pair_index in range(n_pairs):
            u, bound, is_feasible, h = selected_control(
                s[pair_index, time_index],
                v_follower[pair_index, time_index],
                v_leader[pair_index, time_index],
            )
            control[pair_index, time_index] = u
            upper[pair_index, time_index] = bound
            feasible[pair_index, time_index] = is_feasible
            barrier[pair_index, time_index] = h

            if pair_index == 0:
                a_l = platoon.a_leader_original[0, time_index]
            else:
                a_l = control[pair_index - 1, time_index]

            leader_input_valid[pair_index, time_index] = (
                A_MIN - NUM_TOL <= a_l <= A_MAX + NUM_TOL
            )
            delta_v = (
                v_follower[pair_index, time_index]
                - v_leader[pair_index, time_index]
            )
            exact_cbf_residual[pair_index, time_index] = (
                exact_barrier_rate(delta_v, u, a_l) + KAPPA * h
            )

        assumptions_hold_for_platoon = np.all(
            state_domain_valid[:, time_index]
            & feasible[:, time_index]
            & leader_input_valid[:, time_index]
        )
        if formal_prefix_open and assumptions_hold_for_platoon:
            formal_scope[:, time_index] = True
        else:
            formal_prefix_open = False

        if time_index + 1 < n_frames:
            s[:, time_index + 1] = s[:, time_index] + (
                v_leader[:, time_index] - v_follower[:, time_index]
            ) * dt
            v_follower[:, time_index + 1] = (
                v_follower[:, time_index] + control[:, time_index] * dt
            )

    return ControlledData(
        s=s,
        v_follower=v_follower,
        v_leader=v_leader,
        control=control,
        upper=upper,
        feasible=feasible,
        barrier=barrier,
        exact_cbf_residual=exact_cbf_residual,
        leader_input_valid=leader_input_valid,
        active=active,
        speed_domain_valid=speed_domain_valid,
        state_domain_valid=state_domain_valid,
        formal_scope=formal_scope,
        gap_domain_exit=gap_domain_exit,
    )


def temporal_metrics(
    s: np.ndarray, v_i: np.ndarray, v_l: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thw = np.full_like(s, np.nan, dtype=float)
    ttc = np.full_like(s, np.nan, dtype=float)
    delta_v = v_i - v_l

    valid_thw = (s > 0.0) & (v_i > 0.0)
    thw[valid_thw] = s[valid_thw] / v_i[valid_thw]

    valid_ttc = (s > 0.0) & (delta_v > CLOSING_EPS)
    ttc[valid_ttc] = s[valid_ttc] / delta_v[valid_ttc]
    return thw, ttc, delta_v


def count_episodes(flags: np.ndarray) -> int:
    values = np.asarray(flags, dtype=bool)
    if values.size == 0:
        return 0
    starts = values & ~np.concatenate(([False], values[:-1]))
    return int(np.count_nonzero(starts))


def finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else np.nan


def pair_summary_rows(
    platoon_index: int,
    platoon: PlatoonData,
    controlled: ControlledData,
    dt: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    candidate = platoon.candidate
    frames = candidate.frames
    times = (frames - frames[0]) * dt
    n_pairs = platoon.s_original.shape[0]

    h_original = barrier_value(
        platoon.s_original,
        platoon.v_follower_original,
        platoon.v_leader_original,
    )
    thw_original, ttc_original, dv_original = temporal_metrics(
        platoon.s_original,
        platoon.v_follower_original,
        platoon.v_leader_original,
    )
    thw_controlled, ttc_controlled, dv_controlled = temporal_metrics(
        controlled.s,
        controlled.v_follower,
        controlled.v_leader,
    )
    original_speed_domain_valid = (
        (platoon.v_leader_original > SPEED_MIN)
        & (platoon.v_leader_original < SPEED_MAX)
        & (platoon.v_follower_original > SPEED_MIN)
        & (platoon.v_follower_original < SPEED_MAX)
    )
    original_state_domain_valid = (
        (platoon.s_original > 0.0) & original_speed_domain_valid
    )

    records: list[dict[str, object]] = []
    time_series_parts: list[pd.DataFrame] = []
    platoon_gap_exit_by_time = np.any(controlled.gap_domain_exit, axis=0)
    platoon_exit_indices = np.flatnonzero(platoon_gap_exit_by_time)
    first_platoon_exit_index = (
        int(platoon_exit_indices[0]) if platoon_exit_indices.size else None
    )
    first_platoon_exit_frame = (
        int(frames[first_platoon_exit_index])
        if first_platoon_exit_index is not None
        else np.nan
    )
    first_platoon_exit_time = (
        float(times[first_platoon_exit_index])
        if first_platoon_exit_index is not None
        else np.nan
    )

    for pair_index in range(n_pairs):
        leader_id = candidate.vehicle_ids[pair_index]
        follower_id = candidate.vehicle_ids[pair_index + 1]
        original_short = (
            np.isfinite(ttc_original[pair_index])
            & (ttc_original[pair_index] < TAU)
        )
        controlled_short = (
            controlled.active[pair_index]
            & np.isfinite(ttc_controlled[pair_index])
            & (ttc_controlled[pair_index] < TAU)
        )
        controlled_short_formal = (
            controlled.formal_scope[pair_index] & controlled_short
        )
        original_low_thw = (
            np.isfinite(thw_original[pair_index])
            & (thw_original[pair_index] < TAU)
        )
        controlled_low_thw = (
            controlled.active[pair_index]
            & np.isfinite(thw_controlled[pair_index])
            & (thw_controlled[pair_index] < TAU)
        )
        controlled_low_thw_formal = (
            controlled.formal_scope[pair_index] & controlled_low_thw
        )

        assumptions_hold = (
            controlled.active[pair_index]
            & controlled.state_domain_valid[pair_index]
            & controlled.feasible[pair_index]
            & controlled.leader_input_valid[pair_index]
        )
        cbf_violation = assumptions_hold & (
            controlled.exact_cbf_residual[pair_index] < -CBF_TOL
        )
        fallback = controlled.active[pair_index] & (
            ~controlled.feasible[pair_index]
        )
        fallback_indices = np.flatnonzero(fallback)
        first_fallback_index = (
            int(fallback_indices[0]) if fallback_indices.size else None
        )
        first_fallback_frame = (
            int(frames[first_fallback_index])
            if first_fallback_index is not None
            else np.nan
        )
        first_fallback_time = (
            float(times[first_fallback_index])
            if first_fallback_index is not None
            else np.nan
        )
        lower_bound_active = (
            controlled.active[pair_index]
            & np.isclose(
                controlled.control[pair_index], A_MIN, atol=NUM_TOL, rtol=0.0
            )
        )
        upper_bound_active = (
            controlled.active[pair_index]
            & np.isclose(
                controlled.control[pair_index], A_MAX, atol=NUM_TOL, rtol=0.0
            )
        )
        zero_control = (
            controlled.active[pair_index]
            & np.isclose(
                controlled.control[pair_index], 0.0, atol=NUM_TOL, rtol=0.0
            )
        )
        controlled_negative_h = (
            controlled.active[pair_index]
            & (controlled.barrier[pair_index] < 0.0)
        )
        formal_negative_h = (
            controlled.formal_scope[pair_index]
            & (controlled.barrier[pair_index] < 0.0)
        )
        gap_exit_indices = np.flatnonzero(
            controlled.gap_domain_exit[pair_index]
        )
        first_gap_exit_index = (
            int(gap_exit_indices[0]) if gap_exit_indices.size else None
        )
        first_gap_exit_frame = (
            int(frames[first_gap_exit_index])
            if first_gap_exit_index is not None
            else np.nan
        )
        first_gap_exit_time = (
            float(times[first_gap_exit_index])
            if first_gap_exit_index is not None
            else np.nan
        )
        initial_ttc = ttc_original[pair_index, 0]

        records.append(
            {
                "direction": candidate.direction_name,
                "scenario_class": candidate.scenario_class,
                "platoon_index": platoon_index,
                "lane_id": candidate.lane_id,
                "pair_index": pair_index + 1,
                "leader_id": leader_id,
                "follower_id": follower_id,
                "frames": len(frames),
                "duration_s": (len(frames) - 1) * dt,
                "initial_state_domain_valid": bool(
                    original_state_domain_valid[pair_index, 0]
                ),
                "initial_gap_m": float(platoon.s_original[pair_index, 0]),
                "initial_delta_v_mps": float(dv_original[pair_index, 0]),
                "initial_h_m": float(h_original[pair_index, 0]),
                "initial_ttc_s": (
                    float(initial_ttc) if np.isfinite(initial_ttc) else np.nan
                ),
                "initial_implemented_upper_mps2": float(
                    controlled.upper[pair_index, 0]
                ),
                "initial_safe_set": bool(h_original[pair_index, 0] >= -NUM_TOL),
                "initial_controller_feasible": bool(
                    controlled.feasible[pair_index, 0]
                ),
                "initial_leader_input_valid": bool(
                    controlled.leader_input_valid[pair_index, 0]
                ),
                "controller_status": (
                    "encountered_infeasibility"
                    if np.any(fallback)
                    else "remained_feasible"
                ),
                "controlled_active_samples": int(
                    np.count_nonzero(controlled.active[pair_index])
                ),
                "controlled_full_horizon_completed": bool(
                    np.all(controlled.active[pair_index])
                ),
                "controlled_active_duration_s": max(
                    0.0,
                    (np.count_nonzero(controlled.active[pair_index]) - 1) * dt,
                ),
                "formal_scope_samples": int(
                    np.count_nonzero(controlled.formal_scope[pair_index])
                ),
                "original_min_gap_m": float(np.min(platoon.s_original[pair_index])),
                "controlled_min_gap_m": finite_min(controlled.s[pair_index]),
                "original_min_h_m": float(np.min(h_original[pair_index])),
                "controlled_min_h_m": finite_min(controlled.barrier[pair_index]),
                "original_mean_h_m": float(np.mean(h_original[pair_index])),
                "controlled_mean_h_m": finite_mean(controlled.barrier[pair_index]),
                "original_min_thw_s": finite_min(thw_original[pair_index]),
                "controlled_min_thw_s": finite_min(thw_controlled[pair_index]),
                "original_mean_thw_s": finite_mean(thw_original[pair_index]),
                "controlled_mean_thw_s": finite_mean(thw_controlled[pair_index]),
                "original_min_ttc_s": finite_min(ttc_original[pair_index]),
                "controlled_min_ttc_s": finite_min(ttc_controlled[pair_index]),
                "original_mean_ttc_s": finite_mean(ttc_original[pair_index]),
                "controlled_mean_ttc_s": finite_mean(ttc_controlled[pair_index]),
                "original_ttc_active_samples": int(
                    np.count_nonzero(np.isfinite(ttc_original[pair_index]))
                ),
                "controlled_ttc_active_samples": int(
                    np.count_nonzero(
                        controlled.active[pair_index]
                        & np.isfinite(ttc_controlled[pair_index])
                    )
                ),
                "original_short_ttc_samples": int(np.count_nonzero(original_short)),
                "controlled_short_ttc_samples": int(np.count_nonzero(controlled_short)),
                "controlled_formal_scope_short_ttc_samples": int(
                    np.count_nonzero(controlled_short_formal)
                ),
                "original_conflict_episodes": count_episodes(original_short),
                "controlled_conflict_episodes": count_episodes(controlled_short),
                "controlled_formal_scope_conflict_episodes": count_episodes(
                    controlled_short_formal
                ),
                "original_low_thw_samples": int(
                    np.count_nonzero(original_low_thw)
                ),
                "controlled_low_thw_samples": int(
                    np.count_nonzero(controlled_low_thw)
                ),
                "controlled_formal_scope_low_thw_samples": int(
                    np.count_nonzero(controlled_low_thw_formal)
                ),
                "original_low_thw_episodes": count_episodes(original_low_thw),
                "controlled_low_thw_episodes": count_episodes(
                    controlled_low_thw
                ),
                "original_negative_barrier_samples": int(
                    np.count_nonzero(
                        (platoon.s_original[pair_index] > 0.0)
                        & (h_original[pair_index] < 0.0)
                    )
                ),
                "controlled_negative_barrier_samples": int(
                    np.count_nonzero(controlled_negative_h)
                ),
                "controlled_formal_scope_negative_barrier_samples": int(
                    np.count_nonzero(formal_negative_h)
                ),
                "original_invalid_gap_samples": int(
                    np.count_nonzero(platoon.s_original[pair_index] <= 0.0)
                ),
                "controlled_invalid_gap_samples": int(
                    np.count_nonzero(
                        np.isfinite(controlled.s[pair_index])
                        & (controlled.s[pair_index] <= 0.0)
                    )
                ),
                "controlled_gap_domain_exit": bool(gap_exit_indices.size),
                "controlled_first_gap_exit_frame": first_gap_exit_frame,
                "controlled_first_gap_exit_time_s": first_gap_exit_time,
                "platoon_terminated_by_gap": bool(platoon_exit_indices.size),
                "platoon_first_gap_exit_frame": first_platoon_exit_frame,
                "platoon_first_gap_exit_time_s": first_platoon_exit_time,
                "controlled_speed_domain_violation_samples": int(
                    np.count_nonzero(
                        controlled.active[pair_index]
                        & (~controlled.speed_domain_valid[pair_index])
                    )
                ),
                "controller_infeasible_samples": int(
                    np.count_nonzero(fallback)
                ),
                "controller_first_infeasible_frame": first_fallback_frame,
                "controller_first_infeasible_time_s": first_fallback_time,
                "acceleration_at_lower_bound_samples": int(
                    np.count_nonzero(lower_bound_active)
                ),
                "acceleration_at_upper_bound_samples": int(
                    np.count_nonzero(upper_bound_active)
                ),
                "zero_acceleration_samples": int(
                    np.count_nonzero(zero_control)
                ),
                "mean_absolute_selected_acceleration_mps2": finite_mean(
                    np.abs(controlled.control[pair_index][
                        controlled.active[pair_index]
                    ])
                ),
                "leader_input_bound_violations": int(
                    np.count_nonzero(
                        controlled.active[pair_index]
                        & (~controlled.leader_input_valid[pair_index])
                    )
                ),
                "pointwise_cbf_violations_when_assumptions_hold": int(
                    np.count_nonzero(cbf_violation)
                ),
            }
        )

        time_series_parts.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "time_s": times,
                    "direction": candidate.direction_name,
                    "scenario_class": candidate.scenario_class,
                    "platoon_index": platoon_index,
                    "lane_id": candidate.lane_id,
                    "pair_index": pair_index + 1,
                    "leader_id": leader_id,
                    "follower_id": follower_id,
                    "original_gap_m": platoon.s_original[pair_index],
                    "controlled_gap_m": controlled.s[pair_index],
                    "original_leader_speed_mps": platoon.v_leader_original[pair_index],
                    "controlled_leader_speed_mps": controlled.v_leader[pair_index],
                    "original_follower_speed_mps": platoon.v_follower_original[pair_index],
                    "controlled_follower_speed_mps": controlled.v_follower[pair_index],
                    "recorded_follower_acceleration_mps2": platoon.a_follower_original[pair_index],
                    "selected_acceleration_mps2": controlled.control[pair_index],
                    "implemented_upper_mps2": controlled.upper[pair_index],
                    "controller_feasible": controlled.feasible[pair_index],
                    "controller_fallback": fallback,
                    "leader_input_valid": controlled.leader_input_valid[pair_index],
                    "controlled_active": controlled.active[pair_index],
                    "original_speed_domain_valid": original_speed_domain_valid[pair_index],
                    "original_state_domain_valid": original_state_domain_valid[pair_index],
                    "controlled_speed_domain_valid": controlled.speed_domain_valid[pair_index],
                    "controlled_state_domain_valid": controlled.state_domain_valid[pair_index],
                    "controlled_formal_scope": controlled.formal_scope[pair_index],
                    "controlled_gap_domain_exit": controlled.gap_domain_exit[pair_index],
                    "controlled_platoon_gap_domain_exit": platoon_gap_exit_by_time,
                    "original_delta_v_mps": dv_original[pair_index],
                    "controlled_delta_v_mps": dv_controlled[pair_index],
                    "original_barrier_m": h_original[pair_index],
                    "controlled_barrier_m": controlled.barrier[pair_index],
                    "original_thw_s": thw_original[pair_index],
                    "controlled_thw_s": thw_controlled[pair_index],
                    "original_ttc_s": ttc_original[pair_index],
                    "controlled_ttc_s": ttc_controlled[pair_index],
                    "original_short_ttc": original_short,
                    "controlled_short_ttc": controlled_short,
                    "controlled_formal_scope_short_ttc": controlled_short_formal,
                    "original_low_thw": original_low_thw,
                    "controlled_low_thw": controlled_low_thw,
                    "controlled_formal_scope_low_thw": controlled_low_thw_formal,
                    "acceleration_at_lower_bound": lower_bound_active,
                    "acceleration_at_upper_bound": upper_bound_active,
                    "zero_acceleration": zero_control,
                    "exact_pointwise_cbf_residual": controlled.exact_cbf_residual[pair_index],
                }
            )
        )

    return records, pd.concat(time_series_parts, ignore_index=True)


def set_plot_style() -> None:
    # Preserve the typography used in the original publication script.
    plt.rcParams["font.family"] = "Times New Roman"


def save_figure(fig: plt.Figure, stem: Path, show: bool) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_publication_figure(
    fig: plt.Figure, stem: Path, show: bool
) -> None:
    """Save the compact original-style canvas without changing its extent."""
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def plot_publication_platoon(
    direction_index: int,
    platoon: PlatoonData,
    controlled: ControlledData,
    output_dir: Path,
    show: bool,
) -> None:
    """Reproduce the compact five-row platoon layout used in the paper."""
    candidate = platoon.candidate
    n_pairs = platoon.s_original.shape[0]
    sample_index = np.arange(platoon.s_original.shape[1], dtype=int)
    h_original = barrier_value(
        platoon.s_original,
        platoon.v_follower_original,
        platoon.v_leader_original,
    )
    direction_slug = "pos" if candidate.direction_sign > 0 else "neg"
    base_title = (
        f"{candidate.direction_name} Platoon {direction_index} "
        f"({len(candidate.vehicle_ids)} vehicles)"
    )

    # The attached reference image is 1280 x 692 pixels.  At 100 dpi this
    # corresponds to the following canvas while retaining the original fonts,
    # colours, line widths, legends, and two-column organization.
    fig, axes = plt.subplots(
        n_pairs,
        2,
        figsize=(12.8, 6.92),
        dpi=100,
        squeeze=False,
    )
    for pair_index in range(n_pairs):
        leader_id = candidate.vehicle_ids[pair_index]
        follower_id = candidate.vehicle_ids[pair_index + 1]
        pair_title = (
            f"Pair {pair_index + 1}: Leader {leader_id} "
            f"→ Follower {follower_id}"
        )

        gap_axis = axes[pair_index, 0]
        gap_axis.plot(
            sample_index,
            platoon.s_original[pair_index],
            label="Original Gap",
            linewidth=2.5,
        )
        gap_axis.plot(
            sample_index,
            controlled.s[pair_index],
            label="CBF Gap",
            linestyle="--",
            linewidth=2.5,
            color="orange",
        )
        gap_axis.set_ylabel("Gap [m]", fontsize=12)
        gap_axis.set_title(pair_title, fontweight="bold", fontsize=12)
        gap_axis.legend(fontsize=9)
        gap_axis.grid(True)

        barrier_axis = axes[pair_index, 1]
        barrier_axis.plot(
            sample_index,
            h_original[pair_index],
            label="Original h(t)",
            color="red",
            linewidth=2.2,
            alpha=0.9,
        )
        barrier_axis.plot(
            sample_index,
            controlled.barrier[pair_index],
            label="CBF h(t)",
            color="purple",
            linewidth=2.5,
        )
        barrier_axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="$h=0$ boundary",
        )
        barrier_axis.set_ylabel("Barrier h", fontsize=12)
        barrier_axis.set_title(pair_title, fontweight="bold", fontsize=12)
        barrier_axis.legend(fontsize=9)
        barrier_axis.grid(True)

    fig.suptitle(
        f"{base_title} — Gap & Barrier",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_publication_figure(
        fig,
        output_dir / f"{direction_slug}_plat_{direction_index}_gap_h",
        show,
    )

    fig, axes = plt.subplots(
        n_pairs,
        2,
        figsize=(12.8, 6.92),
        dpi=100,
        squeeze=False,
    )
    for pair_index in range(n_pairs):
        leader_id = candidate.vehicle_ids[pair_index]
        follower_id = candidate.vehicle_ids[pair_index + 1]
        pair_title = (
            f"Pair {pair_index + 1}: Leader {leader_id} "
            f"→ Follower {follower_id}"
        )

        speed_axis = axes[pair_index, 0]
        speed_axis.plot(
            sample_index,
            platoon.v_leader_original[pair_index],
            label=f"Leader {leader_id} (Original)",
            color="darkblue",
            linewidth=2.2,
            linestyle="-",
        )
        speed_axis.plot(
            sample_index,
            controlled.v_leader[pair_index],
            label=f"Leader {leader_id} (CBF)",
            color="green",
            linewidth=2.2,
            linestyle="--",
        )
        speed_axis.plot(
            sample_index,
            platoon.v_follower_original[pair_index],
            label=f"Follower {follower_id} (Original)",
            color="orange",
            linewidth=2.2,
            linestyle="-",
        )
        speed_axis.plot(
            sample_index,
            controlled.v_follower[pair_index],
            label=f"Follower {follower_id} (CBF)",
            color="red",
            linewidth=2.2,
            linestyle="--",
        )
        speed_axis.set_ylabel("Speed [m/s]", fontsize=12)
        speed_axis.set_title(pair_title, fontweight="bold", fontsize=12)
        speed_axis.legend(loc="upper left", fontsize=9)
        speed_axis.grid(True)

        control_axis = axes[pair_index, 1]
        control_axis.plot(
            sample_index,
            platoon.a_follower_original[pair_index],
            label=f"Original a(t) - Follower {follower_id}",
            color="blue",
            linewidth=2.0,
            alpha=0.75,
        )
        control_axis.plot(
            sample_index,
            controlled.control[pair_index],
            label=f"CBF u(t) - Follower {follower_id}",
            color="green",
            linewidth=2.0,
            linestyle="--",
        )
        control_axis.axhline(0.0, color="gray", linestyle="--")
        control_axis.axhline(
            A_MIN,
            color="red",
            linestyle=":",
            alpha=0.6,
            label=rf"$a_{{\min}}={A_MIN:g}$",
        )
        control_axis.axhline(
            A_MAX,
            color="red",
            linestyle=":",
            alpha=0.6,
            label=rf"$a_{{\max}}={A_MAX:g}$",
        )
        control_axis.set_ylabel("Acceleration [m/s²]", fontsize=12)
        control_axis.set_title(pair_title, fontweight="bold", fontsize=12)
        control_axis.legend(fontsize=9)
        control_axis.grid(True)

    fig.suptitle(
        f"{base_title} — Speeds & Acceleration",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_publication_figure(
        fig,
        output_dir / f"{direction_slug}_plat_{direction_index}_speed_accel",
        show,
    )


def shade_boolean_intervals(
    axis: plt.Axes,
    times: np.ndarray,
    flags: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.12,
) -> None:
    """Shade contiguous true intervals without duplicating legend entries."""
    indices = np.flatnonzero(flags)
    if indices.size == 0:
        return
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_points)
    for group_index, group in enumerate(groups):
        left = times[group[0]]
        right = times[group[-1]]
        if right <= left and times.size > 1:
            right = left + (times[1] - times[0])
        axis.axvspan(
            left,
            right,
            color=color,
            alpha=alpha,
            label=label if group_index == 0 else None,
        )


def plot_platoon(
    platoon_index: int,
    platoon: PlatoonData,
    controlled: ControlledData,
    dt: float,
    output_dir: Path,
    show: bool,
) -> None:
    candidate = platoon.candidate
    n_pairs = platoon.s_original.shape[0]
    times = np.arange(platoon.s_original.shape[1], dtype=float) * dt
    h_original = barrier_value(
        platoon.s_original,
        platoon.v_follower_original,
        platoon.v_leader_original,
    )
    scenario_label = SCENARIO_LABELS.get(
        candidate.scenario_class, candidate.scenario_class
    )

    fig, axes = plt.subplots(
        n_pairs, 2, figsize=(12, 5 * n_pairs), squeeze=False
    )
    for pair_index in range(n_pairs):
        leader_id = candidate.vehicle_ids[pair_index]
        follower_id = candidate.vehicle_ids[pair_index + 1]
        fallback = controlled.active[pair_index] & (
            ~controlled.feasible[pair_index]
        )
        exit_indices = np.flatnonzero(controlled.gap_domain_exit[pair_index])

        gap_axis = axes[pair_index, 0]
        gap_axis.plot(
            times,
            platoon.s_original[pair_index],
            label="Original Gap",
            linewidth=2.5,
        )
        gap_axis.plot(
            times,
            controlled.s[pair_index],
            label="CBF Gap",
            linestyle="--",
            linewidth=2.5,
            color="orange",
        )
        gap_axis.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
        shade_boolean_intervals(
            gap_axis, times, fallback, "red", "Infeasible fallback"
        )
        if exit_indices.size:
            gap_axis.axvline(
                times[exit_indices[0]],
                color="black",
                linestyle="-.",
                linewidth=1.4,
                label="Gap-domain exit",
            )
        gap_axis.set_ylabel("Gap [m]", fontsize=12)
        gap_axis.set_xlabel("Time [s]")
        gap_axis.set_title(
            f"Pair {pair_index + 1}: Leader {leader_id} → Follower {follower_id}",
            fontweight="bold",
            fontsize=12,
        )
        gap_axis.legend()
        gap_axis.grid(True)

        barrier_axis = axes[pair_index, 1]
        barrier_axis.plot(
            times,
            h_original[pair_index],
            label="Original h(t)",
            color="red",
            linewidth=2.2,
            alpha=0.9,
        )
        barrier_axis.plot(
            times,
            controlled.barrier[pair_index],
            label="CBF h(t)",
            color="purple",
            linewidth=2.5,
        )
        barrier_axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="$h=0$ boundary",
        )
        shade_boolean_intervals(
            barrier_axis, times, fallback, "red", "Infeasible fallback"
        )
        if exit_indices.size:
            barrier_axis.axvline(
                times[exit_indices[0]],
                color="black",
                linestyle="-.",
                linewidth=1.4,
                label="Gap-domain exit",
            )
        barrier_axis.set_ylabel("Barrier h", fontsize=12)
        barrier_axis.set_xlabel("Time [s]")
        barrier_axis.set_title(
            f"Pair {pair_index + 1}: Leader {leader_id} → Follower {follower_id}",
            fontweight="bold",
            fontsize=12,
        )
        barrier_axis.legend()
        barrier_axis.grid(True)

    fig.suptitle(
        f"{candidate.direction_name} Platoon {platoon_index} — "
        f"{scenario_label}: Gap & Barrier",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = output_dir / (
        f"platoon_{platoon_index:02d}_{candidate.direction_name.lower()}_"
        f"{candidate.scenario_class}_gap_barrier"
    )
    save_figure(fig, stem, show)

    fig, axes = plt.subplots(
        n_pairs, 2, figsize=(12, 5 * n_pairs), squeeze=False
    )
    for pair_index in range(n_pairs):
        leader_id = candidate.vehicle_ids[pair_index]
        follower_id = candidate.vehicle_ids[pair_index + 1]
        fallback = controlled.active[pair_index] & (
            ~controlled.feasible[pair_index]
        )
        exit_indices = np.flatnonzero(controlled.gap_domain_exit[pair_index])

        speed_axis = axes[pair_index, 0]
        speed_axis.plot(
            times,
            platoon.v_leader_original[pair_index],
            label=f"Leader {leader_id} (Original)",
            color="darkblue",
            linewidth=2.2,
            linestyle="-",
        )
        speed_axis.plot(
            times,
            controlled.v_leader[pair_index],
            label=f"Leader {leader_id} (CBF)",
            color="green",
            linestyle="--",
            linewidth=2.2,
        )
        speed_axis.plot(
            times,
            platoon.v_follower_original[pair_index],
            label=f"Follower {follower_id} (Original)",
            color="orange",
            linewidth=2.2,
            linestyle="-",
        )
        speed_axis.plot(
            times,
            controlled.v_follower[pair_index],
            label=f"Follower {follower_id} (CBF)",
            color="red",
            linestyle="--",
            linewidth=2.2,
        )
        speed_axis.axhline(SPEED_MIN, color="gray", linestyle=":", linewidth=1.0)
        speed_axis.axhline(SPEED_MAX, color="gray", linestyle=":", linewidth=1.0)
        shade_boolean_intervals(
            speed_axis, times, fallback, "red", "Infeasible fallback"
        )
        if exit_indices.size:
            speed_axis.axvline(
                times[exit_indices[0]], color="black", linestyle="-.", linewidth=1.4
            )
        speed_axis.set_ylabel("Speed [m/s]", fontsize=12)
        speed_axis.set_xlabel("Time [s]")
        speed_axis.set_title(
            f"Pair {pair_index + 1}: Leader {leader_id} → Follower {follower_id}",
            fontweight="bold",
            fontsize=12,
        )
        speed_axis.legend(loc="upper left", fontsize=9)
        speed_axis.grid(True)

        control_axis = axes[pair_index, 1]
        control_axis.plot(
            times,
            platoon.a_follower_original[pair_index],
            label=f"Original a(t) - Follower {follower_id}",
            color="blue",
            linewidth=2.0,
            alpha=0.75,
        )
        control_axis.plot(
            times,
            controlled.control[pair_index],
            label=f"CBF u(t) - Follower {follower_id}",
            color="green",
            linewidth=2.0,
            linestyle="--",
        )
        if np.any(fallback):
            control_axis.scatter(
                times[fallback],
                controlled.control[pair_index, fallback],
                color="red",
                marker="x",
                s=20,
                label="Zero fallback",
                zorder=3,
            )
        shade_boolean_intervals(
            control_axis, times, fallback, "red", "Infeasible interval"
        )
        control_axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        control_axis.axhline(
            A_MIN,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=rf"$a_{{\min}}={A_MIN:g}$",
        )
        control_axis.axhline(
            A_MAX,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=rf"$a_{{\max}}={A_MAX:g}$",
        )
        if exit_indices.size:
            control_axis.axvline(
                times[exit_indices[0]], color="black", linestyle="-.", linewidth=1.4
            )
        control_axis.set_ylabel("Acceleration [m/s²]", fontsize=12)
        control_axis.set_xlabel("Time [s]")
        control_axis.set_title(
            f"Pair {pair_index + 1}: Leader {leader_id} → Follower {follower_id}",
            fontweight="bold",
            fontsize=12,
        )
        control_axis.legend()
        control_axis.grid(True)

    fig.suptitle(
        f"{candidate.direction_name} Platoon {platoon_index} — "
        f"{scenario_label}: Speeds & Acceleration",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = output_dir / (
        f"platoon_{platoon_index:02d}_{candidate.direction_name.lower()}_"
        f"{candidate.scenario_class}_speed_control"
    )
    save_figure(fig, stem, show)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    probabilities = np.linspace(0.0, 1.0, finite.size)
    return finite, probabilities


def plot_safety_cdfs(
    time_series: pd.DataFrame,
    output_dir: Path,
    show: bool,
    stem_name: str = "cdf_safety_metrics",
    title: str = "Distribution of Safety Metrics Across All Pairs",
    thw_display_max: float | None = None,
    ttc_display_max: float | None = None,
    population: str = "formal_scope",
) -> bool:
    if population == "formal_scope":
        comparison_mask = (
            time_series["controlled_formal_scope"].astype(bool)
            & time_series["original_state_domain_valid"].astype(bool)
            & time_series["controlled_state_domain_valid"].astype(bool)
        )
    elif population == "full_controlled":
        comparison_mask = time_series["controlled_active"].astype(bool)
    else:
        raise ValueError(
            "population must be 'formal_scope' or 'full_controlled'."
        )
    comparison = time_series.loc[comparison_mask].copy()
    if comparison.empty:
        population_label = (
            "formal pointwise-CBF scope"
            if population == "formal_scope"
            else "active controlled population"
        )
        print(f"Skipped {stem_name}: no samples lie within the {population_label}.")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specifications = [
        (
            "original_barrier_m",
            "controlled_barrier_m",
            "Barrier $h$ [m]",
            "(a) CDF of Barrier Value",
            0.0,
        ),
        (
            "original_thw_s",
            "controlled_thw_s",
            "Time Headway [s]",
            "(b) CDF of Time Headway",
            TAU,
        ),
        (
            "original_ttc_s",
            "controlled_ttc_s",
            "TTC [s] (during closing)",
            "(c) CDF of TTC",
            TAU,
        ),
    ]

    for plot_index, (axis, specification) in enumerate(zip(
        axes, specifications
    )):
        (
            original_col,
            controlled_col,
            label,
            subplot_title,
            threshold,
        ) = specification
        original_x, original_y = empirical_cdf(
            comparison[original_col].to_numpy(dtype=float)
        )
        controlled_x, controlled_y = empirical_cdf(
            comparison[controlled_col].to_numpy(dtype=float)
        )
        display_limit = (
            thw_display_max
            if plot_index == 1
            else ttc_display_max if plot_index == 2 else None
        )
        panel_notes: list[str] = []
        for values, probabilities, series_label, color in (
            (original_x, original_y, "Original", "red"),
            (controlled_x, controlled_y, "CBF", "green"),
        ):
            if values.size == 0:
                empty_label = series_label
                if plot_index == 2:
                    empty_label += " (TTC inactive)"
                    panel_notes.append(
                        f"{series_label}: no positive-closing samples"
                    )
                else:
                    empty_label += " (no finite samples)"
                axis.plot(
                    [],
                    [],
                    label=empty_label,
                    color=color,
                    linewidth=2.5,
                )
                continue

            plotted_values = values
            if display_limit is not None:
                clipped_count = int(np.count_nonzero(values > display_limit))
                plotted_values = np.minimum(values, display_limit)
                if clipped_count:
                    panel_notes.append(
                        f"{series_label}: {clipped_count} value(s) above "
                        f"{display_limit:g} s shown at the boundary"
                    )
            axis.plot(
                plotted_values,
                probabilities,
                label=series_label,
                color=color,
                linewidth=2.5,
            )
        if plot_index == 0:
            axis.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1.5,
            )
        if plot_index == 1 and thw_display_max is not None:
            axis.set_xlim(0.0, thw_display_max)
        elif plot_index == 2 and ttc_display_max is not None:
            axis.set_xlim(0.0, ttc_display_max)
        if panel_notes:
            axis.text(
                0.02,
                0.04,
                "\n".join(panel_notes),
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": "gray",
                },
            )
        axis.set_xlabel(label, fontweight="bold", fontsize=12)
        if plot_index == 0:
            axis.set_ylabel("CDF")
        axis.set_title(subplot_title, fontweight="bold", fontsize=12)
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / stem_name, show)
    return True


def plot_conflict_counts(
    pair_summary: pd.DataFrame, output_dir: Path, show: bool
) -> None:
    grouped_rows: list[dict[str, object]] = []
    for scenario_class in SCENARIO_CLASSES:
        for direction in ("Positive", "Negative"):
            subset = pair_summary[
                (pair_summary["scenario_class"] == scenario_class)
                & (pair_summary["direction"] == direction)
            ]
            if subset.empty:
                continue
            grouped_rows.append(
                {
                    "label": (
                        f"{SCENARIO_LABELS[scenario_class]}\n{direction}"
                    ),
                    "recorded_samples": int(
                        subset["original_short_ttc_samples"].sum()
                    ),
                    "controlled_samples": int(
                        subset["controlled_short_ttc_samples"].sum()
                    ),
                    "recorded_episodes": int(
                        subset["original_conflict_episodes"].sum()
                    ),
                    "controlled_episodes": int(
                        subset["controlled_conflict_episodes"].sum()
                    ),
                }
            )

    if not grouped_rows:
        return
    grouped = pd.DataFrame(grouped_rows)
    x = np.arange(len(grouped))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, suffix, title in (
        (axes[0], "samples", "Short-TTC samples"),
        (axes[1], "episodes", "Contiguous conflict episodes"),
    ):
        axis.bar(
            x - width / 2,
            grouped[f"recorded_{suffix}"],
            width,
            label="Recorded",
        )
        axis.bar(
            x + width / 2,
            grouped[f"controlled_{suffix}"],
            width,
            label="Controlled, including fallback",
        )
        axis.set_xticks(x, grouped["label"], rotation=20, ha="right")
        axis.set_ylabel("Count")
        axis.set_title(title, fontweight="bold")
        axis.legend()
    fig.suptitle(
        "TTC-Conflict Outcomes by Initial Scenario and Direction",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "ttc_conflict_counts", show)


def plot_safe_start_safety_outcomes(
    pair_summary: pd.DataFrame,
    output_dir: Path,
    show: bool,
) -> None:
    subset = pair_summary[pair_summary["scenario_class"] == SAFE_START]
    if subset.empty:
        return

    sample_recorded = [
        int(subset["original_low_thw_samples"].sum()),
        int(subset["original_short_ttc_samples"].sum()),
    ]
    sample_controlled = [
        int(subset["controlled_low_thw_samples"].sum()),
        int(subset["controlled_short_ttc_samples"].sum()),
    ]
    episode_recorded = [
        int(subset["original_low_thw_episodes"].sum()),
        int(subset["original_conflict_episodes"].sum()),
    ]
    episode_controlled = [
        int(subset["controlled_low_thw_episodes"].sum()),
        int(subset["controlled_conflict_episodes"].sum()),
    ]

    labels = ["THW < $\\tau$", "TTC < $\\tau$"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, recorded, controlled, title, ylabel in (
        (
            axes[0],
            sample_recorded,
            sample_controlled,
            "Threshold-violation samples",
            "Samples",
        ),
        (
            axes[1],
            episode_recorded,
            episode_controlled,
            "Contiguous violation episodes",
            "Episodes",
        ),
    ):
        axis.bar(x - width / 2, recorded, width, label="Recorded")
        axis.bar(
            x + width / 2,
            controlled,
            width,
            label="Controlled, including fallback",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.legend()
    fig.suptitle(
        "Safe-Start TTC--THW Controller Outcomes",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "safe_start_safety_outcomes", show)


def count_grouped_episodes(
    time_series: pd.DataFrame,
    flag_column: str,
    mask: pd.Series,
) -> int:
    total = 0
    keys = ["platoon_index", "pair_index"]
    for _, group in time_series.assign(_mask=mask).groupby(keys, sort=False):
        ordered = group.sort_values("frame")
        flags = (
            ordered[flag_column].astype(bool)
            & ordered["_mask"].astype(bool)
        ).to_numpy(dtype=bool)
        total += count_episodes(flags)
    return total


def formal_scope_summary(
    pair_summary: pd.DataFrame, time_series: pd.DataFrame
) -> pd.DataFrame:
    safe_start_available = np.any(
        (time_series["scenario_class"] == SAFE_START)
        & time_series["controlled_formal_scope"].astype(bool)
    )
    scenario_mask = (
        time_series["scenario_class"] == SAFE_START
        if safe_start_available
        else pd.Series(True, index=time_series.index)
    )
    comparison_mask = (
        scenario_mask
        & time_series["controlled_formal_scope"].astype(bool)
        & time_series["original_state_domain_valid"].astype(bool)
        & time_series["controlled_state_domain_valid"].astype(bool)
    )
    comparison = time_series.loc[comparison_mask]
    scope_name = (
        "safe-start formal-scope samples"
        if safe_start_available
        else "all available formal-scope samples"
    )
    rows: list[dict[str, object]] = []

    def add_value(
        metric: str,
        recorded: float,
        controlled: float,
        unit: str,
        reduction: bool = False,
    ) -> None:
        change = controlled - recorded
        reduction_percent = np.nan
        if reduction and recorded > 0:
            reduction_percent = 100.0 * (recorded - controlled) / recorded
        rows.append(
            {
                "population": scope_name,
                "metric": metric,
                "recorded": recorded,
                "controlled": controlled,
                "change_controlled_minus_recorded": change,
                "reduction_percent": reduction_percent,
                "unit": unit,
            }
        )

    if comparison.empty:
        return pd.DataFrame(rows)

    add_value(
        "Minimum barrier",
        finite_min(comparison["original_barrier_m"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_barrier_m"].to_numpy(dtype=float)),
        "m",
    )
    add_value(
        "Median barrier",
        finite_median(comparison["original_barrier_m"].to_numpy(dtype=float)),
        finite_median(comparison["controlled_barrier_m"].to_numpy(dtype=float)),
        "m",
    )
    add_value(
        "Minimum THW",
        finite_min(comparison["original_thw_s"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_thw_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Median THW",
        finite_median(comparison["original_thw_s"].to_numpy(dtype=float)),
        finite_median(comparison["controlled_thw_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Minimum TTC during closing",
        finite_min(comparison["original_ttc_s"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_ttc_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Fifth-percentile TTC during closing",
        finite_quantile(
            comparison["original_ttc_s"].to_numpy(dtype=float), 0.05
        ),
        finite_quantile(
            comparison["controlled_ttc_s"].to_numpy(dtype=float), 0.05
        ),
        "s",
    )
    add_value(
        "THW-threshold violation samples",
        float(np.count_nonzero(comparison["original_low_thw"].astype(bool))),
        float(np.count_nonzero(comparison["controlled_low_thw"].astype(bool))),
        "samples",
        reduction=True,
    )
    add_value(
        "THW-threshold violation episodes",
        float(count_grouped_episodes(
            time_series, "original_low_thw", comparison_mask
        )),
        float(count_grouped_episodes(
            time_series, "controlled_low_thw", comparison_mask
        )),
        "episodes",
        reduction=True,
    )
    recorded_short = int(
        np.count_nonzero(comparison["original_short_ttc"].astype(bool))
    )
    controlled_short = int(
        np.count_nonzero(comparison["controlled_short_ttc"].astype(bool))
    )
    add_value(
        "Short-TTC samples",
        float(recorded_short),
        float(controlled_short),
        "samples",
        reduction=True,
    )
    add_value(
        "Contiguous TTC-conflict episodes",
        float(count_grouped_episodes(
            time_series, "original_short_ttc", comparison_mask
        )),
        float(count_grouped_episodes(
            time_series, "controlled_short_ttc", comparison_mask
        )),
        "episodes",
        reduction=True,
    )
    add_value(
        "Negative-barrier samples",
        float(np.count_nonzero(comparison["original_barrier_m"] < 0.0)),
        float(np.count_nonzero(comparison["controlled_barrier_m"] < 0.0)),
        "samples",
        reduction=True,
    )
    return pd.DataFrame(rows)


def controller_evaluation_summary(
    pair_summary: pd.DataFrame,
    time_series: pd.DataFrame,
) -> pd.DataFrame:
    """Recorded-versus-controlled outcomes over complete safe-start runs."""
    safe_start = time_series["scenario_class"] == SAFE_START
    comparison_mask = safe_start & time_series["controlled_active"].astype(bool)
    comparison = time_series.loc[comparison_mask]
    rows: list[dict[str, object]] = []

    def add_value(
        metric: str,
        recorded: float,
        controlled: float,
        unit: str,
        reduction: bool = False,
    ) -> None:
        change = controlled - recorded
        reduction_percent = np.nan
        if reduction and recorded > 0:
            reduction_percent = 100.0 * (recorded - controlled) / recorded
        rows.append(
            {
                "population": "full safe-start controlled trajectories",
                "metric": metric,
                "recorded": recorded,
                "controlled": controlled,
                "change_controlled_minus_recorded": change,
                "reduction_percent": reduction_percent,
                "unit": unit,
            }
        )

    if comparison.empty:
        return pd.DataFrame(rows)

    recorded_absolute_acceleration = np.abs(
        comparison["recorded_follower_acceleration_mps2"].to_numpy(
            dtype=float
        )
    )
    controlled_absolute_acceleration = np.abs(
        comparison["selected_acceleration_mps2"].to_numpy(dtype=float)
    )

    add_value(
        "Minimum barrier",
        finite_min(comparison["original_barrier_m"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_barrier_m"].to_numpy(dtype=float)),
        "m",
    )
    add_value(
        "Median barrier",
        finite_median(comparison["original_barrier_m"].to_numpy(dtype=float)),
        finite_median(comparison["controlled_barrier_m"].to_numpy(dtype=float)),
        "m",
    )
    add_value(
        "Minimum THW",
        finite_min(comparison["original_thw_s"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_thw_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Median THW",
        finite_median(comparison["original_thw_s"].to_numpy(dtype=float)),
        finite_median(comparison["controlled_thw_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Minimum TTC during closing",
        finite_min(comparison["original_ttc_s"].to_numpy(dtype=float)),
        finite_min(comparison["controlled_ttc_s"].to_numpy(dtype=float)),
        "s",
    )
    add_value(
        "Fifth-percentile TTC during closing",
        finite_quantile(
            comparison["original_ttc_s"].to_numpy(dtype=float), 0.05
        ),
        finite_quantile(
            comparison["controlled_ttc_s"].to_numpy(dtype=float), 0.05
        ),
        "s",
    )
    add_value(
        "Mean absolute follower acceleration",
        finite_mean(recorded_absolute_acceleration),
        finite_mean(controlled_absolute_acceleration),
        "m/s^2",
    )
    add_value(
        "95th-percentile absolute follower acceleration",
        finite_quantile(recorded_absolute_acceleration, 0.95),
        finite_quantile(controlled_absolute_acceleration, 0.95),
        "m/s^2",
    )
    add_value(
        "THW-threshold violation samples",
        float(np.count_nonzero(comparison["original_low_thw"].astype(bool))),
        float(np.count_nonzero(comparison["controlled_low_thw"].astype(bool))),
        "samples",
        reduction=True,
    )
    add_value(
        "THW-threshold violation episodes",
        float(count_grouped_episodes(
            time_series, "original_low_thw", comparison_mask
        )),
        float(count_grouped_episodes(
            time_series, "controlled_low_thw", comparison_mask
        )),
        "episodes",
        reduction=True,
    )
    add_value(
        "Short-TTC samples",
        float(np.count_nonzero(comparison["original_short_ttc"].astype(bool))),
        float(np.count_nonzero(comparison["controlled_short_ttc"].astype(bool))),
        "samples",
        reduction=True,
    )
    add_value(
        "Contiguous TTC-conflict episodes",
        float(count_grouped_episodes(
            time_series, "original_short_ttc", comparison_mask
        )),
        float(count_grouped_episodes(
            time_series, "controlled_short_ttc", comparison_mask
        )),
        "episodes",
        reduction=True,
    )
    add_value(
        "Negative-barrier samples",
        float(np.count_nonzero(comparison["original_barrier_m"] < 0.0)),
        float(np.count_nonzero(comparison["controlled_barrier_m"] < 0.0)),
        "samples",
        reduction=True,
    )
    return pd.DataFrame(rows)


def scenario_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_class in SCENARIO_CLASSES:
        for direction in ("Positive", "Negative", "All"):
            scenario_subset = pair_summary[
                pair_summary["scenario_class"] == scenario_class
            ]
            subset = (
                scenario_subset
                if direction == "All"
                else scenario_subset[
                    scenario_subset["direction"] == direction
                ]
            )
            if subset.empty:
                continue
            platoon_completion = subset.groupby(
                "platoon_index", sort=False
            )["controlled_full_horizon_completed"].all()
            active_weights = subset["controlled_active_samples"].to_numpy(
                dtype=float
            )
            mean_abs_acceleration = subset[
                "mean_absolute_selected_acceleration_mps2"
            ].to_numpy(dtype=float)
            finite_acceleration = np.isfinite(mean_abs_acceleration)
            weighted_acceleration = np.nan
            if np.any(finite_acceleration):
                weighted_acceleration = float(
                    np.average(
                        mean_abs_acceleration[finite_acceleration],
                        weights=active_weights[finite_acceleration],
                    )
                )
            rows.append(
                {
                    "scenario_class": scenario_class,
                    "direction": direction,
                    "platoons": int(subset["platoon_index"].nunique()),
                    "pairs": int(len(subset)),
                    "platoons_completing_full_horizon": int(
                        np.count_nonzero(platoon_completion)
                    ),
                    "pairs_completing_full_horizon": int(
                        np.count_nonzero(
                            subset["controlled_full_horizon_completed"]
                        )
                    ),
                    "planned_pair_samples": int(subset["frames"].sum()),
                    "controlled_active_samples": int(
                        subset["controlled_active_samples"].sum()
                    ),
                    "formal_scope_samples": int(
                        subset["formal_scope_samples"].sum()
                    ),
                    "controller_infeasible_samples": int(
                        subset["controller_infeasible_samples"].sum()
                    ),
                    "pairs_encountering_infeasibility": int(
                        np.count_nonzero(
                            subset["controller_infeasible_samples"] > 0
                        )
                    ),
                    "pair_gap_domain_exits": int(
                        np.count_nonzero(subset["controlled_gap_domain_exit"])
                    ),
                    "platoons_terminated_by_gap": int(
                        subset.loc[
                            subset["platoon_terminated_by_gap"],
                            "platoon_index",
                        ].nunique()
                    ),
                    "invalid_gap_samples": int(
                        subset["controlled_invalid_gap_samples"].sum()
                    ),
                    "speed_domain_violation_samples": int(
                        subset["controlled_speed_domain_violation_samples"].sum()
                    ),
                    "recorded_low_thw_samples": int(
                        subset["original_low_thw_samples"].sum()
                    ),
                    "controlled_low_thw_samples": int(
                        subset["controlled_low_thw_samples"].sum()
                    ),
                    "recorded_low_thw_episodes": int(
                        subset["original_low_thw_episodes"].sum()
                    ),
                    "controlled_low_thw_episodes": int(
                        subset["controlled_low_thw_episodes"].sum()
                    ),
                    "recorded_ttc_active_samples": int(
                        subset["original_ttc_active_samples"].sum()
                    ),
                    "controlled_ttc_active_samples": int(
                        subset["controlled_ttc_active_samples"].sum()
                    ),
                    "recorded_short_ttc_samples": int(
                        subset["original_short_ttc_samples"].sum()
                    ),
                    "controlled_short_ttc_samples": int(
                        subset["controlled_short_ttc_samples"].sum()
                    ),
                    "recorded_conflict_episodes": int(
                        subset["original_conflict_episodes"].sum()
                    ),
                    "controlled_conflict_episodes": int(
                        subset["controlled_conflict_episodes"].sum()
                    ),
                    "acceleration_at_lower_bound_samples": int(
                        subset["acceleration_at_lower_bound_samples"].sum()
                    ),
                    "zero_acceleration_samples": int(
                        subset["zero_acceleration_samples"].sum()
                    ),
                    "mean_absolute_selected_acceleration_mps2": (
                        weighted_acceleration
                    ),
                    "median_time_to_first_infeasibility_s": finite_median(
                        subset[
                            "controller_first_infeasible_time_s"
                        ].to_numpy(dtype=float)
                    ),
                    "pointwise_cbf_violations_when_assumptions_hold": int(
                        subset[
                            "pointwise_cbf_violations_when_assumptions_hold"
                        ].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def feasibility_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status in ("remained_feasible", "encountered_infeasibility"):
        subset = pair_summary[pair_summary["controller_status"] == status]
        if subset.empty:
            continue
        rows.append(
            {
                "controller_status": status,
                "pairs": int(len(subset)),
                "pairs_completing_full_horizon": int(
                    np.count_nonzero(
                        subset["controlled_full_horizon_completed"]
                    )
                ),
                "planned_pair_samples": int(subset["frames"].sum()),
                "controlled_active_samples": int(
                    subset["controlled_active_samples"].sum()
                ),
                "formal_scope_samples": int(
                    subset["formal_scope_samples"].sum()
                ),
                "controller_infeasible_samples": int(
                    subset["controller_infeasible_samples"].sum()
                ),
                "invalid_gap_samples": int(
                    subset["controlled_invalid_gap_samples"].sum()
                ),
                "recorded_low_thw_samples": int(
                    subset["original_low_thw_samples"].sum()
                ),
                "controlled_low_thw_samples": int(
                    subset["controlled_low_thw_samples"].sum()
                ),
                "recorded_ttc_active_samples": int(
                    subset["original_ttc_active_samples"].sum()
                ),
                "controlled_ttc_active_samples": int(
                    subset["controlled_ttc_active_samples"].sum()
                ),
                "recorded_short_ttc_samples": int(
                    subset["original_short_ttc_samples"].sum()
                ),
                "controlled_short_ttc_samples": int(
                    subset["controlled_short_ttc_samples"].sum()
                ),
                "recorded_conflict_episodes": int(
                    subset["original_conflict_episodes"].sum()
                ),
                "controlled_conflict_episodes": int(
                    subset["controlled_conflict_episodes"].sum()
                ),
                "platoons_terminated_by_gap": int(
                    subset.loc[
                        subset["platoon_terminated_by_gap"], "platoon_index"
                    ].nunique()
                ),
                "pointwise_cbf_violations_when_assumptions_hold": int(
                    subset[
                        "pointwise_cbf_violations_when_assumptions_hold"
                    ].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def print_summary(
    selected: list[PlatoonCandidate],
    pair_summary: pd.DataFrame,
    controller_summary: pd.DataFrame,
    formal_summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> None:
    print("\nSelected evaluation cases")
    print("-" * 80)
    for index, candidate in enumerate(selected, start=1):
        print(
            f"{index:>2}: {candidate.direction_name:<8} lane={candidate.lane_id:<2} "
            f"scenario={candidate.scenario_class:<24} "
            f"vehicles={list(candidate.vehicle_ids)} "
            f"frames={candidate.frames[0]}--{candidate.frames[-1]} "
            f"({candidate.frames.size} samples)"
        )

    def print_metric_table(title: str, table: pd.DataFrame) -> None:
        if table.empty:
            return
        print(f"\n{title}")
        print("-" * 80)
        display = table.copy()
        for column in [
            "recorded",
            "controlled",
            "change_controlled_minus_recorded",
        ]:
            display[column] = display[column].map(lambda value: f"{value:.4g}")
        display["reduction_percent"] = display["reduction_percent"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
        print(display.to_string(index=False))

    print_metric_table(
        "Primary safe-start controller-evaluation summary",
        controller_summary,
    )
    print_metric_table(
        "Supporting pointwise-CBF formal-scope summary",
        formal_summary,
    )

    if not scenarios.empty:
        print("\nScenario and direction summary")
        print("-" * 80)
        print(scenarios.to_string(index=False))

    if not feasibility.empty:
        print("\nFeasibility-stratified outcome summary")
        print("-" * 80)
        print(feasibility.to_string(index=False))

    infeasible = int(pair_summary["controller_infeasible_samples"].sum())
    input_violations = int(pair_summary["leader_input_bound_violations"].sum())
    cbf_violations = int(
        pair_summary["pointwise_cbf_violations_when_assumptions_hold"].sum()
    )
    gap_exits = int(
        np.count_nonzero(pair_summary["controlled_gap_domain_exit"])
    )
    speed_domain_violations = int(
        pair_summary["controlled_speed_domain_violation_samples"].sum()
    )
    print("\nController diagnostics")
    print("-" * 80)
    print(f"Infeasible controller samples: {infeasible}")
    print(f"Pair gap-domain exits: {gap_exits}")
    print(f"Speed-domain violation samples: {speed_domain_violations}")
    print(f"Leader-input bound violations: {input_violations}")
    print(f"Pointwise CBF violations when assumptions hold: {cbf_violations}")


def main() -> None:
    args = parse_args()
    dt = load_dt(args.recording_meta, args.dt)
    if args.analysis_duration <= 0.0:
        raise ValueError("analysis-duration must be strictly positive.")
    if args.publication_platoons_per_direction < 0:
        raise ValueError(
            "publication-platoons-per-direction cannot be negative."
        )
    if args.publication_platoon_size < 2:
        raise ValueError("publication-platoon-size must be at least 2.")
    if args.publication_plot_frames < 2:
        raise ValueError("publication-plot-frames must be at least 2.")
    for limit_name, limit in (
        ("cdf-thw-max", args.cdf_thw_max),
        ("cdf-ttc-max", args.cdf_ttc_max),
    ):
        if limit is not None and limit <= TAU:
            raise ValueError(f"{limit_name} must be greater than tau.")
    window_frames = int(round(args.analysis_duration / dt)) + 1
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    tracks = load_tracks(args.tracks_csv)
    print(
        f"Loaded {len(tracks):,} samples for "
        f"{tracks['id'].nunique():,} vehicles from {args.tracks_csv}."
    )
    print(f"Using dt={dt:.6f} s.")
    print(
        f"Using fixed {args.analysis_duration:.2f} s evaluation windows "
        f"({window_frames} frames)."
    )

    print(
        "Discovering stable vehicle groups from contiguous "
        "leader--follower intervals..."
    )
    candidates = discover_candidates(
        tracks,
        platoon_size=args.platoon_size,
        min_frames=max(args.min_frames, window_frames),
    )
    indexed_tracks = build_track_index(tracks)
    print("Screening initial states for the configured evaluation scenarios...")
    selected = select_scenario_candidates(
        indexed_tracks,
        candidates,
        window_frames=window_frames,
        cases_per_class_direction={
            SAFE_START: args.safe_cases_per_direction,
            CONFLICT_START: args.conflict_cases_per_direction,
            INFEASIBLE_START: args.infeasible_cases_per_direction,
        },
        scan_limit=args.candidate_scan_limit,
        allow_overlap=args.allow_overlap,
    )
    if not selected:
        raise RuntimeError(
            "No qualifying leader--follower scenario window was found. "
            "Confirm that --platoon-size is 2, then reduce "
            "--analysis-duration or increase --candidate-scan-limit."
        )

    publication_rows: list[dict[str, object]] = []
    if args.publication_platoons_per_direction > 0:
        print(
            "Selecting stable multi-vehicle groups for the supplementary "
            "publication-style figures..."
        )
        publication_candidates = discover_candidates(
            tracks,
            platoon_size=args.publication_platoon_size,
            min_frames=max(args.min_frames, args.publication_plot_frames),
        )
        publication_selected = select_publication_platoons(
            publication_candidates,
            plot_frames=args.publication_plot_frames,
            cases_per_direction=args.publication_platoons_per_direction,
        )
        direction_figure_count = {1.0: 0, -1.0: 0}
        for candidate in publication_selected:
            direction_figure_count[candidate.direction_sign] += 1
            direction_index = direction_figure_count[candidate.direction_sign]
            platoon = extract_platoon(indexed_tracks, candidate)
            controlled = simulate_controlled(platoon, dt)
            plot_publication_platoon(
                direction_index,
                platoon,
                controlled,
                output_dir,
                args.show,
            )
            publication_rows.append(
                {
                    "direction": candidate.direction_name,
                    "direction_figure_index": direction_index,
                    "lane_id": candidate.lane_id,
                    "vehicle_ids": " ".join(map(str, candidate.vehicle_ids)),
                    "first_frame": int(candidate.frames[0]),
                    "last_frame": int(candidate.frames[-1]),
                    "num_frames": int(candidate.frames.size),
                    "duration_s": float((candidate.frames.size - 1) * dt),
                    "controller_infeasible_samples": int(
                        np.count_nonzero(controlled.active & ~controlled.feasible)
                    ),
                    "gap_domain_exits": int(
                        np.count_nonzero(controlled.gap_domain_exit)
                    ),
                    "evaluation_role": (
                        "supplementary multi-pair visualization; excluded "
                        "from pairwise aggregate evaluation"
                    ),
                }
            )
        pd.DataFrame(publication_rows).to_csv(
            output_dir / "publication_platoons.csv", index=False
        )

    selected_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    all_time_series: list[pd.DataFrame] = []

    for platoon_index, candidate in enumerate(selected, start=1):
        selected_rows.append(
            {
                "platoon_index": platoon_index,
                "direction": candidate.direction_name,
                "direction_sign": candidate.direction_sign,
                "scenario_class": candidate.scenario_class,
                "selection_score": candidate.selection_score,
                "recorded_challenge_in_window": (
                    candidate.recorded_challenge_in_window
                ),
                "recorded_challenge_samples": (
                    candidate.recorded_challenge_samples
                ),
                "recorded_ttc_conflict_in_window": (
                    candidate.recorded_ttc_conflict_in_window
                ),
                "recorded_ttc_conflict_samples": (
                    candidate.recorded_ttc_conflict_samples
                ),
                "recorded_closing_samples": (
                    candidate.recorded_closing_samples
                ),
                "lane_id": candidate.lane_id,
                "vehicle_ids": " ".join(map(str, candidate.vehicle_ids)),
                "first_frame": int(candidate.frames[0]),
                "last_frame": int(candidate.frames[-1]),
                "num_frames": int(candidate.frames.size),
                "duration_s": float((candidate.frames.size - 1) * dt),
            }
        )

        platoon = extract_platoon(indexed_tracks, candidate)
        controlled = simulate_controlled(platoon, dt)
        rows, time_series = pair_summary_rows(
            platoon_index, platoon, controlled, dt
        )
        pair_rows.extend(rows)
        all_time_series.append(time_series)

        time_series.to_csv(
            output_dir / f"platoon_{platoon_index:02d}_timeseries.csv",
            index=False,
        )
        plot_safety_cdfs(
            time_series,
            output_dir,
            args.show,
            stem_name=f"platoon_{platoon_index:02d}_cdf_safety_metrics",
            title="Distribution of Safety Metrics Across All Pairs",
            thw_display_max=args.cdf_thw_max,
            ttc_display_max=args.cdf_ttc_max,
            population="full_controlled",
        )
        plot_platoon(
            platoon_index,
            platoon,
            controlled,
            dt,
            output_dir,
            args.show,
        )

    selected_df = pd.DataFrame(selected_rows)
    pair_summary = pd.DataFrame(pair_rows)
    time_series = pd.concat(all_time_series, ignore_index=True)
    controller_summary = controller_evaluation_summary(
        pair_summary, time_series
    )
    formal_summary = formal_scope_summary(pair_summary, time_series)
    scenarios = scenario_summary(pair_summary)
    feasibility = feasibility_summary(pair_summary)

    selected_df.to_csv(output_dir / "selected_platoons.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    time_series.to_csv(output_dir / "all_timeseries.csv", index=False)
    controller_summary.to_csv(
        output_dir / "controller_evaluation_summary.csv", index=False
    )
    controller_summary.to_csv(
        output_dir / "aggregate_summary.csv", index=False
    )
    formal_summary.to_csv(
        output_dir / "formal_scope_summary.csv", index=False
    )
    scenarios.to_csv(output_dir / "scenario_summary.csv", index=False)
    feasibility.to_csv(output_dir / "feasibility_summary.csv", index=False)
    time_series.loc[
        time_series["controlled_formal_scope"].astype(bool)
    ].to_csv(output_dir / "formal_scope_timeseries.csv", index=False)

    configuration = {
        "tracks_csv": str(args.tracks_csv.resolve()),
        "recording_meta": (
            str(args.recording_meta.resolve()) if args.recording_meta else None
        ),
        "output_dir": str(output_dir),
        "dt": dt,
        "tau": TAU,
        "kappa": KAPPA,
        "a_min": A_MIN,
        "a_max": A_MAX,
        "closing_epsilon": CLOSING_EPS,
        "platoon_size": args.platoon_size,
        "speed_domain_mps": [SPEED_MIN, SPEED_MAX],
        "analysis_duration_s": args.analysis_duration,
        "window_frames": window_frames,
        "safe_cases_per_direction": args.safe_cases_per_direction,
        "conflict_cases_per_direction": args.conflict_cases_per_direction,
        "infeasible_cases_per_direction": args.infeasible_cases_per_direction,
        "candidate_scan_limit": args.candidate_scan_limit,
        "min_frames": args.min_frames,
        "allow_overlap": args.allow_overlap,
        "cdf_thw_display_max_s": args.cdf_thw_max,
        "cdf_ttc_display_max_s": args.cdf_ttc_max,
        "publication_platoons_per_direction": (
            args.publication_platoons_per_direction
        ),
        "publication_platoon_size": args.publication_platoon_size,
        "publication_plot_frames": args.publication_plot_frames,
        "publication_figure_canvas_inches": [12.8, 6.92],
        "publication_png_dpi": 100,
        "infeasible_fallback_mps2": 0.0,
        "primary_evaluation_definition": (
            "full safe-start controlled trajectories up to the first "
            "nonpositive-gap domain exit, including later infeasibility "
            "and fallback behavior; safe-start cases prioritize later "
            "recorded short-TTC episodes, then other closing or combined-"
            "barrier events, without using the controlled outcome for "
            "selection"
        ),
        "formal_scope_definition": (
            "valid controlled state, feasible controller, bounded leader "
            "input, and uninterrupted prefix from the initial sample"
        ),
    }
    with (output_dir / "run_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)

    safe_start_series = time_series[
        time_series["scenario_class"] == SAFE_START
    ]
    plot_safety_cdfs(
        safe_start_series,
        output_dir,
        args.show,
        stem_name="cdf_safety_metrics",
        title="Distribution of Safety Metrics Across All Pairs",
        thw_display_max=args.cdf_thw_max,
        ttc_display_max=args.cdf_ttc_max,
        population="full_controlled",
    )
    print_summary(
        selected,
        pair_summary,
        controller_summary,
        formal_summary,
        scenarios,
        feasibility,
    )
    print(f"\nSaved corrected results to: {output_dir}")


if __name__ == "__main__":
    main()
