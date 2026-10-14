import csv
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        """Fallback when the optional progress-bar package is unavailable."""
        return iterable

plt.rcParams['font.family'] = 'Times New Roman'
# Constants
DATASET_PATH = "C:\\Users\\Oumai\\OneDrive - Concordia University - Canada\\Ouma-RL\\Traffic_Data\\highD-dataset-v1.0\\data"
TRACK_COUNT = 60
TTC_THRESHOLD = 3.0  # seconds
A_MIN = -6.0         # m/s^2
A_MAX = 3.0          # m/s^2
MIN_SAFE_DISTANCE = 40 # meters
RESULTS_DIR = "ttc_bc_results"
TTC_PLOT_MAX = 6.0
EVENT_PADDING_SECONDS = 1.0
MIN_EVENT_WINDOW_SECONDS = 4.0
CLOSED_LOOP_SUBSTEPS = 10

class HighD_Safety_Analyzer:
    def __init__(self, csv_path, tracks_meta_path, recording_meta_path,
                 background_image_path=None, frame_interval=None, max_frames=None):
        self.csv_path = csv_path
        self.tracks_meta_path = tracks_meta_path
        self.recording_meta_path = recording_meta_path
        self.background_image_path = background_image_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.data = self.load_highd_csv(self.csv_path)
        self.safety_data = []
        self.track_idx = 25
        # Load additional data
        self.static_info = self.load_static_info()
        self.meta_info = self.load_meta_info()
        self.background_image = plt.imread(self.background_image_path) if self.background_image_path else None

        # Validate frame selection parameters
        if frame_interval and max_frames:
            print("Warning: Both frame_interval and max_frames provided. Using frame_interval.")

        if frame_interval:
            if len(frame_interval) != 2:
                raise ValueError("frame_interval must be a tuple of (start_frame, end_frame)")
            if frame_interval[0] >= frame_interval[1]:
                raise ValueError("start_frame must be less than end_frame")

        if max_frames and max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")


    def load_highd_csv(self, path):
        """Load track data from CSV with frame filtering"""
        frames = {}
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row['frame'])

                # Apply frame filtering based on initialization parameters
                if self.frame_interval:
                    start_frame, end_frame = self.frame_interval
                    if frame_num < start_frame or frame_num > end_frame:
                        continue
                elif self.max_frames:
                    # For max_frames, we need to know the minimum frame to establish range
                    # This assumes we're processing sequentially from the beginning
                    if not hasattr(self, '_min_frame_loaded'):
                        self._min_frame_loaded = frame_num
                    if frame_num > self._min_frame_loaded + self.max_frames - 1:
                        continue

                if frame_num not in frames:
                    frames[frame_num] = []
                frames[frame_num].append(row)

        return frames

    def load_static_info(self):
        """Load static vehicle info"""
        static_info = {}
        with open(self.tracks_meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                static_info[int(row['id'])] = row
        return static_info

    def load_meta_info(self):
        """Load recording metadata"""
        with open(self.recording_meta_path, 'r') as f:
            reader = csv.DictReader(f)
            return next(reader)

    @staticmethod
    def _longitudinal_acceleration(vehicle, speed):
        """Return acceleration in the vehicle's direction of travel."""
        raw_acceleration = float(vehicle.get('xAcceleration') or 0.0)
        raw_velocity = float(vehicle.get('xVelocity') or 0.0)
        if raw_velocity < 0.0:
            return -raw_acceleration
        if raw_velocity > 0.0:
            return raw_acceleration
        return 0.0 if speed == 0.0 else raw_acceleration

    @staticmethod
    def _ttc(gap, follower_speed, leader_speed):
        """TTC on the positive-gap, closing domain used by the Lean model."""
        relative_speed = follower_speed - leader_speed
        if gap <= 0.0:
            return float('inf')
        if relative_speed <= 0.0:
            return float('inf')
        return gap / relative_speed

    @staticmethod
    def _cbf_upper_bound(gap, relative_speed, h_value, leader_acceleration):
        """Lean definition U(s,a_L) for alpha(h)=h."""
        return (
            leader_acceleration
            + (h_value - 1.0) * relative_speed ** 2 / gap
        )

    def verify_safety(self, speed_adjustment_factor=None):
        """Apply the acceleration-level TTC-CBF formalized in ``TTC_BC.lean``.
        """
        self.original_ttc_data = []
        self.adjusted_ttc_data = []
        self.safety_data = []
        self.__dict__.pop('_cached_event_windows', None)
        self.__dict__.pop('_cached_closed_loop_rollouts', None)
        self.__dict__.pop('_cached_conflict_recovery_summary', None)

        try:
            frame_rate = float(self.meta_info.get('frameRate', 25.0))
        except (TypeError, ValueError):
            frame_rate = 25.0
        dt = 1.0 / frame_rate if frame_rate > 0.0 else 1.0 / 25.0

        print(f"Analyzing safety for track {self.track_idx} with the Lean TTC-CBF...")

        for frame_num in tqdm(sorted(self.data.keys())):
            vehicles_by_id = {
                int(row['id']): row
                for row in self.data[frame_num]
                if row.get('id')
            }
            for vehicle in self.data[frame_num]:
                try:
                    dhw = float(vehicle.get('dhw') or 0.0)
                    original_speed = abs(float(vehicle['xVelocity']))
                    preceding_id = int(vehicle.get('precedingId') or 0)
                    preceding_vehicle = vehicles_by_id.get(preceding_id)
                    preceding_speed = abs(float(vehicle.get('precedingXVelocity') or 0.0))
                    lane = int(vehicle['laneId'])
                    follower_x = float(vehicle['x'])
                    follower_length = float(vehicle['width'])

                    # highD's dhw is the bumper-to-bumper longitudinal gap, so
                    # it corresponds to spacing = x_L - x_i - L in Lean.
                    gap = dhw
                    relative_speed = original_speed - preceding_speed
                    original_ttc = self._ttc(gap, original_speed, preceding_speed)

                    # Store original data
                    self.original_ttc_data.append({
                        'frame': frame_num,
                        'id': int(vehicle['id']),
                        'ttc': original_ttc,
                        'xVelocity': original_speed,
                        'lane': lane
                    })

                    # The Lean CBF is defined only for a valid positive-gap,
                    # positive-relative-speed leader--follower interaction.
                    valid_state = False
                    if preceding_vehicle is not None:
                        leader_x = float(preceding_vehicle['x'])
                        valid_state = (
                            follower_x > 0.0
                            and leader_x > 0.0
                            and original_speed > 0.0
                            and preceding_speed > 0.0
                            and follower_length > 0.0
                            and gap > 0.0
                            and relative_speed > 0.0
                        )
                    cbf_active = valid_state
                    safe_set = (not cbf_active) or original_ttc >= TTC_THRESHOLD
                    feasible = True
                    cbf_satisfied = True
                    upper_bound = float('inf')
                    nominal_acceleration = self._longitudinal_acceleration(vehicle, original_speed)
                    selected_acceleration = nominal_acceleration
                    leader_acceleration = 0.0

                    if cbf_active:
                        leader_acceleration = self._longitudinal_acceleration(
                            preceding_vehicle, preceding_speed
                        )
                        h_value = original_ttc - TTC_THRESHOLD
                        upper_bound = self._cbf_upper_bound(
                            gap, relative_speed, h_value, leader_acceleration
                        )

                        # Exact executable counterpart of Lean's a_i_star:
                        # max a_min (min a_max (U s a_L)).
                        selected_acceleration = max(A_MIN, min(A_MAX, upper_bound))
                        feasible = A_MIN <= upper_bound
                        cbf_satisfied = feasible and selected_acceleration <= upper_bound

                    original_cbf_residual = float('nan')
                    adjusted_cbf_residual = float('nan')
                    if cbf_active:
                        h_value = original_ttc - TTC_THRESHOLD
                        original_cbf_residual = (
                            -1.0 - gap * (nominal_acceleration - leader_acceleration)
                            / relative_speed ** 2 + h_value
                        )
                        adjusted_cbf_residual = (
                            -1.0 - gap * (selected_acceleration - leader_acceleration)
                            / relative_speed ** 2 + h_value
                        )

                    # One constant-acceleration dataset step under the control
                    # selected by the formal specification.  Both vehicles and
                    # the relative gap are advanced before TTC is recomputed.
                    adjusted_speed = original_speed
                    adjusted_preceding_speed = preceding_speed
                    adjusted_gap = gap
                    if cbf_active:
                        adjusted_speed = max(0.0, original_speed + selected_acceleration * dt)
                        adjusted_preceding_speed = max(
                            0.0, preceding_speed + leader_acceleration * dt
                        )
                        adjusted_gap = max(
                            0.0,
                            gap
                            - relative_speed * dt
                            + 0.5 * (leader_acceleration - selected_acceleration) * dt ** 2
                        )
                    adjusted_ttc = self._ttc(
                        adjusted_gap, adjusted_speed, adjusted_preceding_speed
                    )

                    # Store adjusted data
                    self.adjusted_ttc_data.append({
                        'frame': frame_num,
                        'id': int(vehicle['id']),
                        'ttc': adjusted_ttc,
                        'xVelocity': adjusted_speed,
                        'lane': lane
                    })

                    # Store results
                    self.safety_data.append({
                        'frame': frame_num,
                        'id': int(vehicle['id']),
                        'preceding_id': preceding_id if preceding_id else -1,
                        'lane': lane,
                        'ttc': original_ttc,
                        'adjusted_ttc': adjusted_ttc,
                        'dhw': dhw,
                        'thw': float(vehicle['thw']) if vehicle['thw'] else float('inf'),
                        'safe': safe_set and cbf_satisfied,
                        'safe_set': safe_set,
                        'valid_state': valid_state,
                        'cbf_active': cbf_active,
                        'feasible': feasible,
                        'cbf_satisfied': cbf_satisfied,
                        'cbf_upper_bound': upper_bound,
                        'leaderAcceleration': leader_acceleration,
                        'precedingVelocity': preceding_speed,
                        'nominalAcceleration': nominal_acceleration,
                        'selectedAcceleration': selected_acceleration,
                        'original_cbf_residual': original_cbf_residual,
                        'adjusted_cbf_residual': adjusted_cbf_residual,
                        'x': follower_x,
                        'y': float(vehicle['y']),
                        'xVelocity': adjusted_speed,
                        'originalVelocity': original_speed,
                        'width': follower_length,
                        'height': float(vehicle['height']),
                        'speed_adjusted': cbf_active and not np.isclose(adjusted_speed, original_speed)
                    })
                except (TypeError, ValueError, KeyError):
                    continue

    def _representative_windows(self):
        """Choose one safe-start, fixed-leader short-TTC episode per lane."""
        if hasattr(self, '_cached_event_windows'):
            return self._cached_event_windows
        rate = float(self.meta_info.get('frameRate', 25.0))
        pad = max(1, round(EVENT_PADDING_SECONDS * rate))
        minimum = max(2, round(MIN_EVENT_WINDOW_SECONDS * rate))
        conflicts, all_rows = {}, {}
        for row in self.safety_data:
            key = (row['lane'], row['id'], row['preceding_id'])
            all_rows.setdefault(key, []).append(row)
            if row.get('cbf_active') and np.isfinite(row['ttc']) and row['ttc'] < TTC_THRESHOLD:
                conflicts.setdefault(key, []).append(row)

        candidates = []
        for (lane, vehicle_id, preceding_id), rows in conflicts.items():
            rows.sort(key=lambda row: row['frame'])
            episodes, episode = [], [rows[0]]
            for row in rows[1:]:
                if row['frame'] == episode[-1]['frame'] + 1:
                    episode.append(row)
                else:
                    episodes.append(episode)
                    episode = [row]
            episodes.append(episode)
            for episode in episodes:
                pair_rows = sorted(all_rows[(lane, vehicle_id, preceding_id)],
                                   key=lambda row: row['frame'])
                event_start = episode[0]['frame']
                event_end = episode[-1]['frame']
                event_index = next(
                    index for index, row in enumerate(pair_rows)
                    if row['frame'] == event_start)
                segment_start_index = event_index
                while (segment_start_index > 0
                       and pair_rows[segment_start_index]['frame']
                       == pair_rows[segment_start_index - 1]['frame'] + 1):
                    segment_start_index -= 1
                segment_end_index = event_index
                while (segment_end_index + 1 < len(pair_rows)
                       and pair_rows[segment_end_index + 1]['frame']
                       == pair_rows[segment_end_index]['frame'] + 1):
                    segment_end_index += 1
                segment_rows = pair_rows[segment_start_index:segment_end_index + 1]
                safe_context = [
                    row for row in segment_rows
                    if row['frame'] < event_start
                    and row.get('cbf_active')
                    and np.isfinite(row['ttc'])
                    and row['ttc'] >= TTC_THRESHOLD
                ]
                # Forward invariance in the Lean theorem assumes SafeSet at
                # initialization.  Episodes without a safe starting sample
                # are therefore unsuitable as controller demonstrations.
                if not safe_context:
                    continue
                candidates.append({
                    'lane': lane, 'vehicle_id': vehicle_id,
                    'preceding_id': preceding_id,
                    'event_start': event_start, 'event_end': event_end,
                    'segment_start': segment_rows[0]['frame'],
                    'segment_end': segment_rows[-1]['frame'],
                    'conflict_frames': len(episode),
                    'min_ttc': min(row['ttc'] for row in episode),
                })

        windows = {}
        for item in candidates:
            score = (item['conflict_frames'], -item['min_ttc'])
            if item['lane'] not in windows or score > windows[item['lane']]['score']:
                windows[item['lane']] = dict(item, score=score)
        for window in windows.values():
            rows = sorted(all_rows[(window['lane'], window['vehicle_id'],
                                    window['preceding_id'])],
                          key=lambda row: row['frame'])
            rows = [row for row in rows
                    if window['segment_start'] <= row['frame'] <= window['segment_end']]
            available_start = window['segment_start']
            available_end = window['segment_end']
            start = max(available_start, window['event_start'] - pad)
            end = min(available_end, window['event_end'] + pad)
            if end - start + 1 < minimum:
                center = (window['event_start'] + window['event_end']) // 2
                start = max(available_start, center - minimum // 2)
                end = min(available_end, start + minimum - 1)
                start = max(available_start, end - minimum + 1)
            row_by_frame = {row['frame']: row for row in rows}
            if (start not in row_by_frame
                    or not np.isfinite(row_by_frame[start]['ttc'])
                    or row_by_frame[start]['ttc'] < TTC_THRESHOLD):
                safe_starts = [
                    row['frame'] for row in rows
                    if row['frame'] < window['event_start']
                    and np.isfinite(row['ttc']) and row['ttc'] >= TTC_THRESHOLD
                ]
                start = max(safe_starts)
            window.update(plot_start=start, plot_end=end)
            window.pop('segment_start')
            window.pop('segment_end')
            window.pop('score')
        self._cached_event_windows = dict(sorted(windows.items()))
        return self._cached_event_windows

    def _event_rows(self, window):
        return sorted([
            row for row in self.safety_data
            if row['lane'] == window['lane'] and row['id'] == window['vehicle_id']
            and row['preceding_id'] == window['preceding_id']
            and window['plot_start'] <= row['frame'] <= window['plot_end']
        ], key=lambda row: row['frame'])

    def _closed_loop_rollout(self, window):
        """Recursively propagate one follower against its recorded leader.

        The follower speed, leader speed, and relative gap at sample k become
        the state at sample k+1.  Recorded leader acceleration is the external
        input.  Ten internal integration steps reduce discretization error
        relative to the continuous-time Lean CBF condition.
        """
        if not hasattr(self, '_cached_closed_loop_rollouts'):
            self._cached_closed_loop_rollouts = {}
        key = (window['lane'], window['vehicle_id'], window['preceding_id'],
               window['plot_start'], window['plot_end'])
        if key in self._cached_closed_loop_rollouts:
            return self._cached_closed_loop_rollouts[key]

        rows = self._event_rows(window)
        if not rows:
            return []
        rate = float(self.meta_info.get('frameRate', 25.0))
        base_dt = 1.0 / rate if rate > 0.0 else 1.0 / 25.0
        follower_speed = rows[0]['originalVelocity']
        leader_speed_state = rows[0]['precedingVelocity']
        gap = rows[0]['dhw']
        initial_safe = (np.isfinite(rows[0]['ttc'])
                        and rows[0]['ttc'] >= TTC_THRESHOLD)
        rollout = []

        for index, row in enumerate(rows):
            leader_speed = leader_speed_state
            leader_acceleration = row['leaderAcceleration']
            relative_speed = follower_speed - leader_speed
            controlled_ttc = self._ttc(gap, follower_speed, leader_speed)
            h_value = controlled_ttc - TTC_THRESHOLD
            active = (gap > 0.0 and follower_speed > 0.0 and leader_speed > 0.0
                      and relative_speed > 0.0 and np.isfinite(controlled_ttc))
            upper_bound = float('inf')
            feasible = True
            command = max(A_MIN, min(A_MAX, row['nominalAcceleration']))
            if active:
                upper_bound = self._cbf_upper_bound(
                    gap, relative_speed, h_value, leader_acceleration)
                command = max(A_MIN, min(A_MAX, upper_bound))
                feasible = A_MIN <= upper_bound

            rollout.append({
                'frame': row['frame'], 'lane': row['lane'], 'id': row['id'],
                'preceding_id': row['preceding_id'],
                'recorded_ttc': row['ttc'], 'controlled_ttc': controlled_ttc,
                'recorded_speed': row['originalVelocity'],
                'controlled_speed': follower_speed,
                'controlled_gap': gap, 'selected_acceleration': command,
                'cbf_upper_bound': upper_bound, 'feasible': feasible,
                'initial_safe': initial_safe,
            })

            if index == len(rows) - 1:
                continue
            frame_step = max(1, rows[index + 1]['frame'] - row['frame'])
            internal_dt = base_dt * frame_step / CLOSED_LOOP_SUBSTEPS
            sub_leader_speed = leader_speed
            for _ in range(CLOSED_LOOP_SUBSTEPS):
                relative_speed = follower_speed - sub_leader_speed
                sub_ttc = self._ttc(gap, follower_speed, sub_leader_speed)
                if (gap > 0.0 and follower_speed > 0.0 and sub_leader_speed > 0.0
                        and relative_speed > 0.0 and np.isfinite(sub_ttc)):
                    sub_h = sub_ttc - TTC_THRESHOLD
                    sub_upper = self._cbf_upper_bound(
                        gap, relative_speed, sub_h, leader_acceleration)
                    command = max(A_MIN, min(A_MAX, sub_upper))
                else:
                    command = max(A_MIN, min(A_MAX, row['nominalAcceleration']))
                gap = max(0.0, gap - relative_speed * internal_dt
                          + 0.5 * (leader_acceleration - command) * internal_dt ** 2)
                follower_speed = max(0.0, follower_speed + command * internal_dt)
                sub_leader_speed = max(
                    0.0, sub_leader_speed + leader_acceleration * internal_dt)
            leader_speed_state = sub_leader_speed

        self._cached_closed_loop_rollouts[key] = rollout
        return rollout

    def _plot_event_metric(self, plot_type, filename):
        windows = self._representative_windows()
        if not windows:
            print(f'No short-TTC episodes available for {filename}.')
            return
        rate = float(self.meta_info.get('frameRate', 25.0))
        n_rows = (len(windows) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.8 * n_rows), squeeze=False)
        axes = list(axes.flat)
        for ax, window in zip(axes, windows.values()):
            rows = self._event_rows(window)
            rollout = self._closed_loop_rollout(window)
            time = np.array([(row['frame'] - window['plot_start']) / rate for row in rows])
            title = (f"Lane {window['lane']} — Vehicle {window['vehicle_id']} "
                     f"(frames {window['plot_start']}–{window['plot_end']})")
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.set_xlabel('Time within selected window (s)', fontsize=14)
            if len(time):
                ax.set_xlim(0.0, max(time[-1], 1.0 / rate))
            ax.grid(True, alpha=0.3, linestyle='--')

            if plot_type in ('ttc', 'original_ttc'):
                original = np.array([row['ttc'] for row in rows], dtype=float)
                visible = np.isfinite(original) & (original <= TTC_PLOT_MAX)
                ax.set_ylabel('TTC (s)', fontsize=14)
                ax.set_ylim(0, TTC_PLOT_MAX)
                if plot_type == 'original_ttc':
                    conflict = visible & (original < TTC_THRESHOLD)
                    ax.plot(time[visible], original[visible], color='#d97706', linewidth=2.2,
                            label='Original TTC')
                    ax.scatter(time[conflict], original[conflict], color='#b91c1c', s=18,
                               zorder=3, label='Short-TTC samples')
                else:
                    adjusted = np.array([row['controlled_ttc'] for row in rollout], dtype=float)
                    original[~visible] = np.nan
                    adjusted[~np.isfinite(adjusted) | (adjusted > TTC_PLOT_MAX)] = np.nan
                    ax.plot(time, original, '--', color='#9e1b1b', linewidth=2,
                            label='Original TTC')
                    ax.plot(time, adjusted, color='#1f77b4', linewidth=2.2,
                            label='Recursive CBF rollout')
                ax.axhline(TTC_THRESHOLD, color='black', linestyle=':', linewidth=1.5,
                           label='3-s threshold')
            else:
                original = np.array([row['originalVelocity'] * 3.6 for row in rows])
                adjusted = np.array([row['controlled_speed'] * 3.6 for row in rollout])
                ax.set_ylabel('Speed (km/h)', fontsize=14)
                ax.plot(time, original, '--', color='#6b7280', linewidth=2,
                        label='Original speed')
                ax.plot(time, adjusted, color='#2563eb', linewidth=2.2,
                        label='Recursive CBF speed')
                ax.fill_between(time, original, adjusted, color='#93c5fd', alpha=0.35)
                low, high = min(original.min(), adjusted.min()), max(original.max(), adjusted.max())
                padding = max(0.5, 0.08 * max(high - low, 1.0))
                ax.set_ylim(max(0, low - padding), high + padding)
            ax.legend(fontsize=10, loc='best')
        for ax in axes[len(windows):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _lane_summary(self):
        summary = []
        recovery = {row['lane']: row for row in self._conflict_recovery_summary()}
        for lane in sorted(set(row['lane'] for row in self.safety_data)):
            rows = [row for row in self.safety_data if row['lane'] == lane]
            active = [row for row in rows if row.get('cbf_active')]
            window = self._representative_windows().get(lane)
            rollout = self._closed_loop_rollout(window) if window else []
            finite_controlled = [row['controlled_ttc'] for row in rollout
                                 if np.isfinite(row['controlled_ttc'])]
            summary.append({
                'lane': lane,
                'short_ttc_before': sum(row['ttc'] < TTC_THRESHOLD for row in rows),
                'short_ttc_after_one_step': sum(row['adjusted_ttc'] < TTC_THRESHOLD for row in rows),
                'selected_window_recorded_short_ttc': sum(
                    row['ttc'] < TTC_THRESHOLD for row in self._event_rows(window)
                ) if window else 0,
                'selected_window_rollout_short_ttc': sum(
                    row['controlled_ttc'] < TTC_THRESHOLD for row in rollout
                ),
                'selected_window_min_rollout_ttc': min(finite_controlled, default=float('inf')),
                'selected_window_infeasible_steps': sum(not row['feasible'] for row in rollout),
                'conflict_onset_recorded_short_ttc': recovery.get(lane, {}).get(
                    'recorded_short_ttc', 0),
                'conflict_onset_rollout_short_ttc': recovery.get(lane, {}).get(
                    'rollout_short_ttc', 0),
                'conflict_episodes': recovery.get(lane, {}).get('episodes', 0),
                'cbf_violations_recorded_control': sum(row['original_cbf_residual'] < 0 for row in active),
                'cbf_violations_filtered_control': sum(row['adjusted_cbf_residual'] < -1e-9 for row in active),
                'infeasible_states': sum(not row.get('feasible', True) for row in active),
            })
        return summary

    def _conflict_recovery_summary(self):
        """Roll out the CBF from each recorded conflict onset.

        Unlike the safe-start representative trajectories, these experiments
        deliberately initialize at the first TTC < 3 s sample.  They measure
        mitigation/recovery, not the forward-invariance guarantee, and thus
        retain genuine post-control short-TTC samples.
        """
        if hasattr(self, '_cached_conflict_recovery_summary'):
            return self._cached_conflict_recovery_summary
        conflicts = {}
        for row in self.safety_data:
            if (row.get('cbf_active') and np.isfinite(row['ttc'])
                    and row['ttc'] < TTC_THRESHOLD):
                key = (row['lane'], row['id'], row['preceding_id'])
                conflicts.setdefault(key, []).append(row)

        by_lane = {}
        for (lane, vehicle_id, preceding_id), rows in conflicts.items():
            rows.sort(key=lambda row: row['frame'])
            episodes, episode = [], [rows[0]]
            for row in rows[1:]:
                if row['frame'] == episode[-1]['frame'] + 1:
                    episode.append(row)
                else:
                    episodes.append(episode)
                    episode = [row]
            episodes.append(episode)
            lane_result = by_lane.setdefault(lane, {
                'lane': lane, 'recorded_short_ttc': 0,
                'rollout_short_ttc': 0, 'episodes': 0,
                'infeasible_rollout_steps': 0,
            })
            for episode in episodes:
                window = {
                    'lane': lane, 'vehicle_id': vehicle_id,
                    'preceding_id': preceding_id,
                    'plot_start': episode[0]['frame'],
                    'plot_end': episode[-1]['frame'],
                }
                rollout = self._closed_loop_rollout(window)
                lane_result['recorded_short_ttc'] += len(episode)
                lane_result['rollout_short_ttc'] += sum(
                    row['controlled_ttc'] < TTC_THRESHOLD for row in rollout)
                lane_result['episodes'] += 1
                lane_result['infeasible_rollout_steps'] += sum(
                    not row['feasible'] for row in rollout)
        self._cached_conflict_recovery_summary = [
            by_lane[lane] for lane in sorted(by_lane)
        ]
        return self._cached_conflict_recovery_summary

    def _plot_cbf_condition_violations(self):
        summary = self._lane_summary()
        lanes = np.array([row['lane'] for row in summary])
        before = np.array([row['cbf_violations_recorded_control'] for row in summary])
        after = np.array([row['cbf_violations_filtered_control'] for row in summary])
        x = np.arange(len(lanes))
        fig, (ax_before, ax_after) = plt.subplots(
            2, 1, figsize=(11, 8), sharex=True,
            gridspec_kw={'height_ratios': [2.6, 1.4]})
        b1 = ax_before.bar(x, before, 0.58, color='#9b1c1c',
                           label=r'Recorded acceleration: $\dot{h}+h<0$')
        b2 = ax_after.bar(x, after, 0.58, color='#1f8a2b',
                          label=r'Lean-CBF acceleration: $\dot{h}+h<0$')
        ax_before.bar_label(b1, padding=3, fontsize=10)
        ax_after.bar_label(b2, padding=3, fontsize=10)
        ax_after.set_ylim(0, max(1, after.max()) * 1.25)
        ax_after.set_xticks(x)
        ax_after.set_xticklabels([int(lane) for lane in lanes])
        ax_after.set_xlabel('Lane ID', fontsize=13)
        ax_before.set_ylabel('Recorded violations', fontsize=12)
        ax_after.set_ylabel('Residual violations', fontsize=12)
        fig.suptitle(r'Violations of the Verified Condition $\dot{h}+h\geq0$',
                     fontsize=15, fontweight='bold')
        for ax in (ax_before, ax_after):
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'cbf_condition_violations.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_selection_and_summary(self):
        windows = self._representative_windows()
        with open(os.path.join(RESULTS_DIR, 'selected_event_windows.csv'), 'w', newline='') as output:
            fields = ['lane','vehicle_id','preceding_id','event_start','event_end',
                      'plot_start','plot_end','conflict_frames','min_ttc']
            writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
            writer.writerows(windows.values())
        rollout_fields = [
            'frame','lane','id','preceding_id','recorded_ttc','controlled_ttc',
            'recorded_speed','controlled_speed','controlled_gap',
            'selected_acceleration','cbf_upper_bound','feasible','initial_safe'
        ]
        with open(os.path.join(RESULTS_DIR, 'closed_loop_event_rollouts.csv'),
                  'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=rollout_fields)
            writer.writeheader()
            for window in windows.values():
                writer.writerows(self._closed_loop_rollout(window))
        summary = self._lane_summary()
        if summary:
            with open(os.path.join(RESULTS_DIR, 'lane_summary.csv'), 'w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(summary[0])); writer.writeheader()
                writer.writerows(summary)

    def _plot_ttc(self, original_ttc_data, adjusted_ttc_data):
        """Visualize TTC adjustments with 2-column grid layout by lane - only vehicles with TTC changes"""
        # Organize data by lane
        lane_groups = {}
        for data in original_ttc_data + adjusted_ttc_data:
            lane = data.get('lane')
            if lane is not None:
                if lane not in lane_groups:
                    lane_groups[lane] = {'original': [], 'adjusted': []}

        # Populate lane groups with data
        for data in original_ttc_data:
            lane = data.get('lane')
            if lane in lane_groups:
                lane_groups[lane]['original'].append(data)

        for data in adjusted_ttc_data:
            lane = data.get('lane')
            if lane in lane_groups:
                lane_groups[lane]['adjusted'].append(data)

        # Get lanes with both original and adjusted data
        lanes_with_data = []
        for lane, data in lane_groups.items():
            if data['original'] and data['adjusted']:
                lanes_with_data.append(lane)

        if not lanes_with_data:
            print("No TTC data available for any lane")
            return

        lanes_with_data.sort()

        # Create subplots in 2-column grid
        n_lanes = len(lanes_with_data)
        n_cols = 2
        n_rows = (n_lanes + 1) // 2  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

        # Flatten axes array for easy iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        # Color palette for different vehicles
        vehicle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, lane in enumerate(lanes_with_data):
            if idx >= len(axes_flat):
                break  # Safety check

            ax = axes_flat[idx]
            lane_data = lane_groups[lane]

            # Get unique vehicles in this lane
            all_vehicles = set()
            for data in lane_data['original']:
                if 'id' in data:
                    all_vehicles.add(data['id'])
            for data in lane_data['adjusted']:
                if 'id' in data:
                    all_vehicles.add(data['id'])

            unique_vehicles = sorted(all_vehicles)
            vehicles_with_ttc_changes = []  # Only plot vehicles with actual TTC changes

            all_ttc_values = []  # Collect all TTC values for dynamic y-axis limits

            for veh_id in unique_vehicles:
                # Get original TTC values for this vehicle in this lane
                orig_ttc = sorted([(d['frame'], d.get('ttc', 0))
                                for d in lane_data['original']
                                if d.get('id') == veh_id and 0 <= d.get('ttc', 0) <= 20])

                # Get adjusted TTC values for this vehicle in this lane
                adj_ttc = sorted([(d['frame'], d.get('ttc', 0))
                                for d in lane_data['adjusted']
                                if d.get('id') == veh_id and 0 <= d.get('ttc', 0) <= 20])

                # Only include vehicles where TTC actually changed
                if orig_ttc and adj_ttc:
                    # Check if TTC values are different between original and adjusted
                    orig_values = [ov for _, ov in orig_ttc]
                    adj_values = [av for _, av in adj_ttc]

                    # Only plot if there are actual differences in TTC
                    if orig_values != adj_values:
                        vehicles_with_ttc_changes.append(veh_id)

            # Limit to 5 vehicles per lane for clarity
            vehicles_to_plot = vehicles_with_ttc_changes[:2]

            if not vehicles_to_plot:
                ax.text(0.5, 0.5, f'No TTC changes in Lane {lane}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'Lane {lane} - No TTC Changes', fontsize=12, fontweight='bold')
                continue

            for i, veh_id in enumerate(vehicles_to_plot):
                # Get original TTC values for this vehicle in this lane
                orig_ttc = sorted([(d['frame'], d.get('ttc', 0))
                                for d in lane_data['original']
                                if d.get('id') == veh_id and 0 <= d.get('ttc', 0) <= 35])

                # Get adjusted TTC values for this vehicle in this lane
                adj_ttc = sorted([(d['frame'], d.get('ttc', 0))
                                for d in lane_data['adjusted']
                                if d.get('id') == veh_id and 0 <= d.get('ttc', 0) <= 35])

                if orig_ttc and adj_ttc:
                    # Extract frames and values
                    orig_frames, orig_values = zip(*orig_ttc)
                    adj_frames, adj_values = zip(*adj_ttc)

                    # Collect TTC values for y-axis limits
                    all_ttc_values.extend(orig_values)
                    all_ttc_values.extend(adj_values)

                    line_style = ['-', '--', '-.', ':', '-'][i % 5]
                    color = vehicle_colors[i % len(vehicle_colors)]

                    # Plot original TTC (dashed line)
                    ax.plot(orig_frames, orig_values, linestyle=line_style, color=color,
                        alpha=0.7, linewidth=3.0,
                        label=f'Veh {veh_id} (Original)')

                    # Plot adjusted TTC (solid line)
                    ax.plot(adj_frames, adj_values, linestyle=line_style, color=color,
                        alpha=1.0, linewidth=2.5,
                        label=f'Veh {veh_id} (Adjusted)')

                    # Add arrows showing direction of change at each point
                    #for (of, ov), (af, av) in zip(orig_ttc, adj_ttc):
                        #if of == af:  # Only where frames match
                            #ax.annotate('', xy=(af, av), xytext=(of, ov),
                                    #arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1.2))

            # Formatting for each subplot
            ax.set_title(f'Lane {lane} - TTC Comparison', fontsize=20, fontweight='bold', fontname='Times New Roman')
            ax.set_xlabel('Frame Number', fontsize=20, fontname='Times New Roman')
            ax.set_ylabel('TTC (seconds)', fontsize=20, fontname='Times New Roman')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Set dynamic y-axis limits based on actual TTC values
            if all_ttc_values:
                y_min = 0
                y_max = max(all_ttc_values) * 1.1  # Add 10% padding
                ax.set_ylim(y_min, y_max)

            # Add critical threshold
            ax.axhline(y=3, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Critical (3s)')

            # Create comprehensive legend with all vehicle entries
            handles, labels = ax.get_legend_handles_labels()

            # Remove duplicate labels while preserving order (keep first occurrence)
            seen = set()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in seen:
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)


            ax.legend(handles=unique_handles, labels=unique_labels,
                    loc='lower right', fontsize=10, framealpha=0.9, ncol=2)

        # Hide any unused subplots
        for idx in range(len(lanes_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Overall figure title
        #fig.suptitle('TTC Comparison: Original vs Speed-Adjusted Values by Lane',
                    #fontsize=16, fontweight='bold', y=0.99)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.6, wspace=0.1)  # Adjust for suptitle
        fig.savefig(os.path.join(RESULTS_DIR, "ttc_original_vs_adjusted.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_dhw_over_time(self):
        """Plot distance headway evolution"""
        fig = plt.figure(figsize=(14, 5))
        vehicle_subset = sorted(set(d['id'] for d in self.safety_data))[:10]

        for vid in vehicle_subset:
            v_data = sorted([d for d in self.safety_data if d['id'] == vid], key=lambda x: x['frame'])
            frames = [d['frame'] for d in v_data]
            dhw = [d['dhw'] for d in v_data]
            unsafe = [d['frame'] for d in v_data if not d['safe']]
            unsafe_dhw = [d['dhw'] for d in v_data if not d['safe']]

            plt.plot(frames, dhw, label=f'Vehicle {vid}')
            plt.scatter(unsafe, unsafe_dhw, color='red', marker='x', s=50)

        plt.axhline(MIN_SAFE_DISTANCE, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Distance Headway (First {15331-1+1} Frames)')
        plt.xlabel('Frame Number')
        plt.ylabel('Distance (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, "distance_headway_over_time.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_ttc_original(self, original_ttc_data):
        """Visualize original TTC values with 2-column grid layout by lane"""
        # Organize data by lane
        lane_groups = {}
        for data in original_ttc_data:
            lane = data.get('lane')
            if lane is not None:
                if lane not in lane_groups:
                    lane_groups[lane] = []
                lane_groups[lane].append(data)

        # Get lanes with TTC data
        lanes_with_data = [lane for lane, data in lane_groups.items() if data]

        if not lanes_with_data:
            print("No TTC data available for any lane")
            return

        lanes_with_data.sort()

        # Create subplots in 2-column grid
        n_lanes = len(lanes_with_data)
        n_cols = 2
        n_rows = (n_lanes + 1) // 2  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

        # Flatten axes array for easy iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        # Color palette for different vehicles
        vehicle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for idx, lane in enumerate(lanes_with_data):
            if idx >= len(axes_flat):
                break  # Safety check

            ax = axes_flat[idx]
            lane_data = lane_groups[lane]

            # Get unique vehicles in this lane
            unique_vehicles = list(set(data.get('id') for data in lane_data if data.get('id') is not None))
            unique_vehicles.sort()

            # Limit to 5 vehicles per lane for clarity
            vehicles_to_plot = unique_vehicles[:3]

            all_ttc_values = []  # Collect all TTC values for dynamic y-axis limits

            for i, veh_id in enumerate(vehicles_to_plot):
                # Get original TTC values for this vehicle in this lane
                ttc_data = sorted([(d['frame'], d.get('ttc', 0))
                                for d in lane_data
                                if d.get('id') == veh_id and 0 <= d.get('ttc', 0) <= 3])

                if ttc_data:
                    # Extract frames and values
                    frames, values = zip(*ttc_data)

                    # Collect TTC values for y-axis limits
                    all_ttc_values.extend(values)

                    color = vehicle_colors[i % len(vehicle_colors)]

                    # Plot original TTC (solid line)
                    ax.plot(frames, values, linestyle='-', color=color,
                        alpha=1.0, linewidth=2.5, marker='o', markersize=4,
                        label=f'Vehicle {veh_id}')

            # Formatting for each subplot
            ax.set_title(f'Lane {lane} - Original TTC Values', fontsize=20, fontweight='bold', fontname='Times New Roman')
            ax.set_xlabel('Frame Number', fontsize=20, fontname='Times New Roman')
            ax.set_ylabel('TTC (seconds)', fontsize=20, fontname='Times New Roman')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Set dynamic y-axis limits based on actual TTC values
            if all_ttc_values:
                y_min = 0
                y_max = max(all_ttc_values) * 1.1  # Add 10% padding
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0, 10)  # Default range if no data

            # Add critical threshold
            ax.axhline(y=3, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Critical (3s)')

            # Create legend with vehicle entries
            handles, labels = ax.get_legend_handles_labels()

            # Remove duplicate labels while preserving order
            seen = set()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in seen:
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)

            # Add critical threshold to legend if not already present
            critical_patch = plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7,
                                    linewidth=1.5, label='Critical (3s)')
            if 'Critical (3s)' not in unique_labels:
                unique_handles.append(critical_patch)
                unique_labels.append('Critical (3s)')

            ax.legend(handles=unique_handles, labels=unique_labels,
                    loc='upper right', fontsize=10, framealpha=0.9, ncol=2)

        # Hide any unused subplots
        for idx in range(len(lanes_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Overall figure title
        #fig.suptitle('Original TTC Values by Lane',
                    #fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.42, wspace=0.15)
        fig.savefig(os.path.join(RESULTS_DIR, "ttc_original_by_lane.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_safety_violations(self):
        """Analyze safety violation patterns by lane."""

        # Create a single figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # --- Data Preparation: By Lane ---
        # Get sorted unique lane IDs
        lanes = sorted(set(d['lane'] for d in self.safety_data))

        violations = [
            sum(1 for d in self.safety_data if not d['safe'] and d['lane'] == lane)
            for lane in lanes
        ]

        # --- Plotting ---
        # Create the bar chart and store the bar container
        bars = ax.bar(lanes, violations, color=plt.cm.tab10.colors[:len(lanes)])

        # Add the violation count above each bar
        ax.bar_label(bars, padding=3)

        # --- Aesthetics ---
        ax.set_title('Safety Violations by Lane', fontsize=14)
        ax.set_xlabel('Lane ID', fontsize=12)
        ax.set_ylabel('Unsafe Interactions', fontsize=12)

        # Add a horizontal grid for readability
        ax.grid(axis='y', alpha=0.5)

        # Overall Title
        #plt.suptitle('Safety Violations Analysis: By Lane')
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "safety_violations_by_lane.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_speed_adjustments(self):
        """Visualize speed adjustments with 2-column grid layout"""
        adjusted_vehicles = [v for v in self.safety_data if v.get('speed_adjusted')]
        if not adjusted_vehicles:
            print("No speed adjustments were needed")
            return

        # Group vehicles by lane
        lane_groups = {}
        for v in adjusted_vehicles:
            lane = v.get('lane')
            if lane is not None:
                if lane not in lane_groups:
                    lane_groups[lane] = []
                lane_groups[lane].append(v)

        # Create subplots in 2-column grid
        lanes = sorted(lane_groups.keys())
        n_lanes = len(lanes)

        if n_lanes == 0:
            print("No lanes with speed adjustments found")
            return

        # Calculate grid dimensions: 2 columns, rows = ceil(n_lanes / 2)
        n_cols = 2
        n_rows = (n_lanes + 1) // 2  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

        # Flatten axes array for easy iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        # Color palette for different vehicles within same lane
        vehicle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, lane in enumerate(lanes):
            if idx >= len(axes_flat):
                break  # Safety check

            ax = axes_flat[idx]
            vehicles = lane_groups[lane]

            # Get unique vehicles in this lane (limit to 5 per lane for clarity)
            unique_vehicles = list(set(v.get('id') for v in vehicles if v.get('id') is not None))[:3]

            all_speeds = []  # Collect speeds for this lane's y-axis limits

            for i, vid in enumerate(unique_vehicles):
                if vid is None:
                    continue

                v_data = [d for d in vehicles if d.get('id') == vid]
                v_data = sorted(v_data, key=lambda x: x.get('frame', 0))

                frames = [d.get('frame', 0) for d in v_data]

                # Always use positive speed values for consistent plotting
                original = [abs(d.get('originalVelocity', 0)) * 3.6 for d in v_data]
                adjusted = [abs(d.get('xVelocity', 0)) * 3.6 for d in v_data]

                # Add to all_speeds for y-axis limits
                all_speeds.extend(original)
                all_speeds.extend(adjusted)

                # Different line styles for different vehicles
                line_style = ['-', '--', '-.', ':', '-'][i % 5]
                color = vehicle_colors[i % len(vehicle_colors)]

                # Plot original speed (lighter)
                ax.plot(frames, original, linestyle=line_style, color=color,
                    alpha=0.6, linewidth=1.5,
                    label=f'Vehicle {vid} (Original)')

                # Plot adjusted speed (darker, thicker)
                ax.plot(frames, adjusted, linestyle=line_style, color=color,
                    alpha=1.0, linewidth=2.5,
                    label=f'Vehicle {vid} (Adjusted)')

                # Add arrows showing direction of change at each point
                #for frame, orig_speed, adj_speed in zip(frames, original, adjusted):
                    #if abs(adj_speed - orig_speed) > 0.1:
                        #ax.annotate('', xy=(frame, adj_speed), xytext=(frame, orig_speed),
                                #arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1))

            # Formatting for each subplot
            ax.set_title(f'Lane {lane} - Speed Adjustments', fontsize=20, fontweight='bold', fontname='Times New Roman')
            ax.set_xlabel('Frame Number', fontsize=20, fontname='Times New Roman')
            ax.set_ylabel('Speed (km/h)', fontsize=20, fontname='Times New Roman')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

            # Set y-axis limits for this lane's plot
            if all_speeds:
                y_min = max(0, min(all_speeds) - 5)
                y_max = max(all_speeds) + 5
                ax.set_ylim(y_min, y_max)

        # Hide any unused subplots
        for idx in range(len(lanes), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Overall figure title
        #fig.suptitle('Speed Adjustments by Lane', fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.42, wspace=0.15)  # Adjust for suptitle
        fig.savefig(os.path.join(RESULTS_DIR, "speed_adjustments_by_lane.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_conflict_comparison(self):
        """Compare conflict-onset recovery for every short-TTC episode."""
        recovery = self._conflict_recovery_summary()
        conflicts_before = {row['lane']: row['recorded_short_ttc'] for row in recovery}
        conflicts_after = {row['lane']: row['rollout_short_ttc'] for row in recovery}
        all_lanes = sorted(conflicts_before)
        if not all_lanes:
            print("No conflicts found before or after adjustment for comparison.")
            return

        # Prepare data for plotting
        lanes = np.array(all_lanes)
        before_counts = np.array([conflicts_before.get(lane, 0) for lane in all_lanes])
        after_counts = np.array([conflicts_after.get(lane, 0) for lane in all_lanes])

        # Plotting
        x = np.arange(len(lanes))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, before_counts, width,
                        label='Recorded trajectory', color='#8B0000')
        rects2 = ax.bar(x + width/2, after_counts, width,
                        label='CBF rollout from conflict onset', color='#228B22')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of Critical Conflicts ($TTC < 3.0$s)', fontsize=10)
        ax.set_xlabel('Lane ID', fontsize=10)
        ax.set_title('Short-TTC Samples During Conflict-Onset Recovery',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lanes)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Function to attach a text label above each bar
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate('{}'.format(int(height)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "conflict_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_results(self):
        """Generate safety analysis plots"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self._plot_event_metric('ttc', 'ttc_original_vs_adjusted.png')
        self._plot_event_metric('original_ttc', 'ttc_original_by_lane.png')
        #self._plot_ttc_per_lane_direction()
        #self._plot_dhw_over_time()
        self._plot_cbf_condition_violations()
        self._plot_event_metric('speed', 'speed_adjustments_by_lane.png')
        self._plot_conflict_comparison()
        self._save_selection_and_summary()
        print(f"Plots saved to: {os.path.abspath(RESULTS_DIR)}")

    def _plot_ttc_per_lane_direction(self):
        """Plot TTC values per lane, separated by direction"""
        positive_lanes = [6, 7, 8]  # Right direction (positive speed)
        negative_lanes = [2, 3, 4]  # Left direction (negative speed)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle('TTC Values per Lane by Direction', y=1.02)

        for i, lane in enumerate(positive_lanes):
            ax = axes[0, i]
            self._plot_lane_ttc(ax, lane, f'Right Direction - Lane {lane}', False)

        for i, lane in enumerate(negative_lanes):
            ax = axes[1, i]
            self._plot_lane_ttc(ax, lane, f'Left Direction - Lane {lane}', True)

        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "ttc_by_lane_direction.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run_full_analysis(self, speed_adjustment_factor=0.8):
        """Complete analysis pipeline"""
        self.verify_safety(speed_adjustment_factor)

if __name__ == '__main__':
    print("Running Lean TTC-CBF analysis on highD Track 25...")
    track_idx = 25
    track_prefix = f"{track_idx:02d}"
    analyzer = HighD_Safety_Analyzer(
        csv_path=os.path.join(DATASET_PATH, f"{track_prefix}_tracks.csv"),
        tracks_meta_path=os.path.join(DATASET_PATH, f"{track_prefix}_tracksMeta.csv"),
        recording_meta_path=os.path.join(DATASET_PATH, f"{track_prefix}_recordingMeta.csv"),
        background_image_path=os.path.join(DATASET_PATH, f"{track_prefix}_highway.png")
    )
    analyzer.track_idx = track_idx
    analyzer.run_full_analysis()

    print("Generating TTC, safety-violation, speed, and conflict plots...")
    analyzer.plot_results()
