import Mathlib

namespace CBFProject

noncomputable section

/-!
This file formalizes pointwise properties of the combined TTC--THW barrier.
It does not formalize trajectories, differentiate an ODE, or prove forward
invariance.  `barrierRate` is the supplied piecewise directional-rate model
used in the algebraic CBF condition.
-/

/-! ## State, input, and parameters -/

/-- Longitudinal state of a leader and its follower. -/
structure VehicleState where
  x_L : ℝ
  v_L : ℝ
  x_i : ℝ
  v_i : ℝ
  L   : ℝ

/-- The leader acceleration is the exogenous input to the follower controller. -/
structure VehicleInput where
  a_L : ℝ

/-- Common TTC--THW threshold. -/
def τ : ℝ := 2.5

/-- Linear CBF gain used by the implementation. -/
def κ : ℝ := 1

/-- Permitted longitudinal-acceleration interval. -/
def a_min : ℝ := -8
def a_max : ℝ := 3

lemma τ_pos : (0 : ℝ) < τ := by
  norm_num [τ]

lemma κ_pos : (0 : ℝ) < κ := by
  norm_num [κ]

lemma two_τ_pos : (0 : ℝ) < 2 * τ := by
  norm_num [τ]

lemma a_min_le_zero : a_min ≤ (0 : ℝ) := by
  norm_num [a_min]

lemma zero_le_a_max : (0 : ℝ) ≤ a_max := by
  norm_num [a_max]

lemma a_min_le_a_max : a_min ≤ a_max := by
  norm_num [a_min, a_max]

/-- Bumper-to-bumper spacing. -/
def spacing (state : VehicleState) : ℝ :=
  state.x_L - state.x_i - state.L

/-- Relative speed, positive when the follower closes on the leader. -/
def relSpeed (state : VehicleState) : ℝ :=
  state.v_i - state.v_L

/-- State domain used for the traffic interpretation. -/
def ValidState (state : VehicleState) : Prop :=
  0 < state.x_L ∧
  0 < state.x_i ∧
  0 < state.L ∧
  0 < spacing state ∧
  5 < state.v_L ∧ state.v_L < 40 ∧
  5 < state.v_i ∧ state.v_i < 40

/-- Only the exogenous leader acceleration is constrained here. -/
def ValidInput (input : VehicleInput) : Prop :=
  a_min ≤ input.a_L ∧ input.a_L ≤ a_max

/-- The permitted interval for a selected follower acceleration. -/
def PermittedAcceleration (a_i : ℝ) : Prop :=
  a_min ≤ a_i ∧ a_i ≤ a_max

lemma spacing_pos (state : VehicleState) (hV : ValidState state) :
    0 < spacing state := by
  exact hV.2.2.2.1

lemma follower_speed_pos (state : VehicleState) (hV : ValidState state) :
    0 < state.v_i := by
  linarith [hV.2.2.2.2.2.2.1]

lemma leader_acceleration_lower_bound (input : VehicleInput)
    (hU : ValidInput input) : a_min ≤ input.a_L := by
  exact hU.1

/-! ## TTC, THW, and the combined barrier -/

/-- Time headway.  Its traffic interpretation uses `state.v_i > 0`. -/
def THW (state : VehicleState) : ℝ :=
  spacing state / state.v_i

/-- Time to collision.  Its traffic interpretation uses positive spacing and relative speed. -/
def TTC (state : VehicleState) : ℝ :=
  spacing state / relSpeed state

/-- Combined TTC--THW barrier. -/
def ttcThwBarrier (state : VehicleState) : ℝ :=
  spacing state - τ * state.v_i - τ * max (relSpeed state) 0

/-- Safe-set predicate, interpreted only together with `ValidState`. -/
def SafeSet (state : VehicleState) : Prop :=
  0 ≤ ttcThwBarrier state

lemma barrier_of_nonpositive_relSpeed (state : VehicleState)
    (hΔv : relSpeed state ≤ 0) :
    ttcThwBarrier state = spacing state - τ * state.v_i := by
  unfold ttcThwBarrier
  rw [max_eq_right hΔv]
  ring

lemma barrier_of_positive_relSpeed (state : VehicleState)
    (hΔv : 0 < relSpeed state) :
    ttcThwBarrier state =
      spacing state - τ * (state.v_i + relSpeed state) := by
  unfold ttcThwBarrier
  rw [max_eq_left (le_of_lt hΔv)]
  ring

/-- In the non-closing regime, the barrier is exactly the THW threshold condition. -/
theorem safe_iff_thw_of_nonclosing (state : VehicleState)
    (hV : ValidState state) (hΔv : relSpeed state ≤ 0) :
    SafeSet state ↔ τ ≤ THW state := by
  have hv : 0 < state.v_i := follower_speed_pos state hV
  constructor
  · intro hSafe
    have hGap : τ * state.v_i ≤ spacing state := by
      unfold SafeSet at hSafe
      rw [barrier_of_nonpositive_relSpeed state hΔv] at hSafe
      linarith
    exact (le_div_iff₀ hv).2 hGap
  · intro hTHW
    have hGap : τ * state.v_i ≤ spacing state :=
      (le_div_iff₀ hv).1 hTHW
    unfold SafeSet
    rw [barrier_of_nonpositive_relSpeed state hΔv]
    linarith

/-- In the closing regime, the safe set is the combined spacing inequality. -/
theorem safe_iff_combined_gap_of_closing (state : VehicleState)
    (hΔv : 0 < relSpeed state) :
    SafeSet state ↔
      τ * (state.v_i + relSpeed state) ≤ spacing state := by
  unfold SafeSet
  rw [barrier_of_positive_relSpeed state hΔv]
  constructor <;> intro h <;> linarith

/-- The combined closing condition implies both individual spacing inequalities. -/
theorem closing_safe_implies_individual_gaps (state : VehicleState)
    (hV : ValidState state) (hΔv : 0 < relSpeed state)
    (hSafe : SafeSet state) :
    τ * state.v_i ≤ spacing state ∧
      τ * relSpeed state ≤ spacing state := by
  have hCombined :
      τ * (state.v_i + relSpeed state) ≤ spacing state :=
    (safe_iff_combined_gap_of_closing state hΔv).1 hSafe
  have hτΔv : 0 < τ * relSpeed state := mul_pos τ_pos hΔv
  have hτvi : 0 < τ * state.v_i :=
    mul_pos τ_pos (follower_speed_pos state hV)
  constructor <;> nlinarith

/-- A safe closing state satisfies the THW threshold. -/
theorem closing_safe_implies_thw (state : VehicleState)
    (hV : ValidState state) (hΔv : 0 < relSpeed state)
    (hSafe : SafeSet state) :
    τ ≤ THW state := by
  have hGap := (closing_safe_implies_individual_gaps state hV hΔv hSafe).1
  exact (le_div_iff₀ (follower_speed_pos state hV)).2 hGap

/-- A safe closing state satisfies the TTC threshold. -/
theorem closing_safe_implies_ttc (state : VehicleState)
    (hV : ValidState state) (hΔv : 0 < relSpeed state)
    (hSafe : SafeSet state) :
    τ ≤ TTC state := by
  have hGap := (closing_safe_implies_individual_gaps state hV hΔv hSafe).2
  exact (le_div_iff₀ hΔv).2 hGap

/-! ## Supplied piecewise barrier-rate model -/

/--
The supplied directional-rate expression.  At zero relative speed, the
directional rate of the positive-part term is `max (a_i - a_L) 0`.
-/
def barrierRate (state : VehicleState) (input : VehicleInput) (a_i : ℝ) : ℝ :=
  if 0 < relSpeed state then
    -relSpeed state - τ * a_i - τ * (a_i - input.a_L)
  else if relSpeed state < 0 then
    -relSpeed state - τ * a_i
  else
    -τ * a_i - τ * max (a_i - input.a_L) 0

/-- Pointwise CBF inequality for the supplied rate. -/
def PointwiseCBF (state : VehicleState) (input : VehicleInput) (a_i : ℝ) : Prop :=
  0 ≤ barrierRate state input a_i + κ * ttcThwBarrier state

lemma barrierRate_of_positive_relSpeed (state : VehicleState)
    (input : VehicleInput) (a_i : ℝ) (hΔv : 0 < relSpeed state) :
    barrierRate state input a_i =
      -relSpeed state - τ * a_i - τ * (a_i - input.a_L) := by
  simp [barrierRate, hΔv]

lemma barrierRate_of_negative_relSpeed (state : VehicleState)
    (input : VehicleInput) (a_i : ℝ) (hΔv : relSpeed state < 0) :
    barrierRate state input a_i = -relSpeed state - τ * a_i := by
  have hNotPos : ¬ 0 < relSpeed state := not_lt.mpr (le_of_lt hΔv)
  simp [barrierRate, hNotPos, hΔv]

lemma barrierRate_of_zero_relSpeed (state : VehicleState)
    (input : VehicleInput) (a_i : ℝ) (hΔv : relSpeed state = 0) :
    barrierRate state input a_i =
      -τ * a_i - τ * max (a_i - input.a_L) 0 := by
  simp [barrierRate, hΔv]

/-! ## Exact branch-specific safe-control conditions -/

def upperPositive (state : VehicleState) (input : VehicleInput) : ℝ :=
  (-relSpeed state + τ * input.a_L + κ * ttcThwBarrier state) / (2 * τ)

def upperNegative (state : VehicleState) : ℝ :=
  (-relSpeed state + κ * ttcThwBarrier state) / τ

def SafePositive (state : VehicleState) (input : VehicleInput) (a_i : ℝ) : Prop :=
  a_i ≤ upperPositive state input

def SafeNegative (state : VehicleState) (a_i : ℝ) : Prop :=
  a_i ≤ upperNegative state

/-- Exact directional-rate condition at `relSpeed state = 0`. -/
def SafeZero (state : VehicleState) (input : VehicleInput) (a_i : ℝ) : Prop :=
  0 ≤ -τ * a_i - τ * max (a_i - input.a_L) 0 +
    κ * ttcThwBarrier state

def SafeControl (state : VehicleState) (input : VehicleInput) (a_i : ℝ) : Prop :=
  if 0 < relSpeed state then
    SafePositive state input a_i
  else if relSpeed state < 0 then
    SafeNegative state a_i
  else
    SafeZero state input a_i

/-- Below the leader acceleration, the zero-speed condition has the non-closing form. -/
theorem safeZero_iff_of_follower_le_leader_acceleration
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hAcc : a_i ≤ input.a_L) :
    SafeZero state input a_i ↔
      a_i ≤ κ * ttcThwBarrier state / τ := by
  have hSub : a_i - input.a_L ≤ 0 := sub_nonpos.mpr hAcc
  unfold SafeZero
  rw [max_eq_right hSub]
  constructor
  · intro h
    apply (le_div_iff₀ τ_pos).2
    nlinarith
  · intro h
    have hMul := (le_div_iff₀ τ_pos).1 h
    nlinarith

/-- Above the leader acceleration, the zero-speed condition has the closing form. -/
theorem safeZero_iff_of_leader_acceleration_lt_follower
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hAcc : input.a_L < a_i) :
    SafeZero state input a_i ↔
      a_i ≤ (τ * input.a_L + κ * ttcThwBarrier state) / (2 * τ) := by
  have hSub : 0 ≤ a_i - input.a_L := sub_nonneg.mpr (le_of_lt hAcc)
  unfold SafeZero
  rw [max_eq_left hSub]
  constructor
  · intro h
    apply (le_div_iff₀ two_τ_pos).2
    nlinarith
  · intro h
    have hMul := (le_div_iff₀ two_τ_pos).1 h
    nlinarith

theorem safePositive_implies_pointwise_cbf
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : 0 < relSpeed state) (hSafe : SafePositive state input a_i) :
    PointwiseCBF state input a_i := by
  unfold SafePositive upperPositive at hSafe
  have hMul := (le_div_iff₀ two_τ_pos).1 hSafe
  unfold PointwiseCBF
  rw [barrierRate_of_positive_relSpeed state input a_i hΔv]
  nlinarith

theorem safeNegative_implies_pointwise_cbf
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state < 0) (hSafe : SafeNegative state a_i) :
    PointwiseCBF state input a_i := by
  unfold SafeNegative upperNegative at hSafe
  have hMul := (le_div_iff₀ τ_pos).1 hSafe
  unfold PointwiseCBF
  rw [barrierRate_of_negative_relSpeed state input a_i hΔv]
  nlinarith

theorem safeZero_implies_pointwise_cbf
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state = 0) (hSafe : SafeZero state input a_i) :
    PointwiseCBF state input a_i := by
  unfold SafeZero at hSafe
  unfold PointwiseCBF
  rw [barrierRate_of_zero_relSpeed state input a_i hΔv]
  exact hSafe

theorem pointwise_cbf_iff_safePositive
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : 0 < relSpeed state) :
    PointwiseCBF state input a_i ↔ SafePositive state input a_i := by
  constructor
  · intro hCBF
    unfold PointwiseCBF at hCBF
    rw [barrierRate_of_positive_relSpeed state input a_i hΔv] at hCBF
    unfold SafePositive upperPositive
    apply (le_div_iff₀ two_τ_pos).2
    nlinarith
  · exact safePositive_implies_pointwise_cbf state input a_i hΔv

theorem pointwise_cbf_iff_safeNegative
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state < 0) :
    PointwiseCBF state input a_i ↔ SafeNegative state a_i := by
  constructor
  · intro hCBF
    unfold PointwiseCBF at hCBF
    rw [barrierRate_of_negative_relSpeed state input a_i hΔv] at hCBF
    unfold SafeNegative upperNegative
    apply (le_div_iff₀ τ_pos).2
    nlinarith
  · exact safeNegative_implies_pointwise_cbf state input a_i hΔv

theorem pointwise_cbf_iff_safeZero
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state = 0) :
    PointwiseCBF state input a_i ↔ SafeZero state input a_i := by
  constructor
  · intro hCBF
    unfold PointwiseCBF at hCBF
    rw [barrierRate_of_zero_relSpeed state input a_i hΔv] at hCBF
    exact hCBF
  · exact safeZero_implies_pointwise_cbf state input a_i hΔv

/-- Every exact branch-specific safe control satisfies the pointwise CBF inequality. -/
theorem safe_control_implies_pointwise_cbf
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hSafe : SafeControl state input a_i) :
    PointwiseCBF state input a_i := by
  by_cases hPos : 0 < relSpeed state
  · have hBranch : SafePositive state input a_i := by
      simpa [SafeControl, hPos] using hSafe
    exact safePositive_implies_pointwise_cbf state input a_i hPos hBranch
  · by_cases hNeg : relSpeed state < 0
    · have hBranch : SafeNegative state a_i := by
        simpa [SafeControl, hPos, hNeg] using hSafe
      exact safeNegative_implies_pointwise_cbf state input a_i hNeg hBranch
    · have hZero : relSpeed state = 0 := by
        exact le_antisymm (le_of_not_gt hPos) (le_of_not_gt hNeg)
      have hBranch : SafeZero state input a_i := by
        simpa [SafeControl, hPos, hNeg] using hSafe
      exact safeZero_implies_pointwise_cbf state input a_i hZero hBranch

/-! ## Exact branch feasibility -/

def FeasiblePositive (state : VehicleState) (input : VehicleInput) : Prop :=
  a_min ≤ upperPositive state input

/-- Exact feasibility condition for the negative-relative-speed branch. -/
def FeasibleNegative (state : VehicleState) : Prop :=
  a_min ≤ upperNegative state

/--
Exact feasibility condition at zero relative speed.  Because the zero-branch
left-hand side is nonincreasing in `a_i`, feasibility is characterized at the
smallest permitted follower acceleration.
-/
def FeasibleZero (state : VehicleState) (input : VehicleInput) : Prop :=
  SafeZero state input a_min

theorem feasiblePositive_iff_exists_bounded_control
    (state : VehicleState) (input : VehicleInput) :
    FeasiblePositive state input ↔
      ∃ a_i : ℝ,
        PermittedAcceleration a_i ∧ SafePositive state input a_i := by
  constructor
  · intro hF
    refine ⟨a_min, ⟨le_rfl, a_min_le_a_max⟩, ?_⟩
    simpa [FeasiblePositive, SafePositive] using hF
  · rintro ⟨a_i, hBounds, hSafe⟩
    unfold FeasiblePositive SafePositive at *
    exact le_trans hBounds.1 hSafe

theorem feasibleNegative_iff_exists_bounded_control
    (state : VehicleState) :
    FeasibleNegative state ↔
      ∃ a_i : ℝ,
        PermittedAcceleration a_i ∧ SafeNegative state a_i := by
  constructor
  · intro hF
    refine ⟨a_min, ⟨le_rfl, a_min_le_a_max⟩, ?_⟩
    simpa [FeasibleNegative, SafeNegative] using hF
  · rintro ⟨a_i, hBounds, hSafe⟩
    unfold FeasibleNegative SafeNegative at *
    exact le_trans hBounds.1 hSafe

theorem feasibleZero_iff_exists_bounded_control
    (state : VehicleState) (input : VehicleInput) :
    FeasibleZero state input ↔
      ∃ a_i : ℝ,
        PermittedAcceleration a_i ∧ SafeZero state input a_i := by
  constructor
  · intro hF
    exact ⟨a_min, ⟨le_rfl, a_min_le_a_max⟩, hF⟩
  · rintro ⟨a_i, hBounds, hSafe⟩
    have hSub : a_min - input.a_L ≤ a_i - input.a_L := by
      linarith [hBounds.1]
    have hMax :
        max (a_min - input.a_L) 0 ≤ max (a_i - input.a_L) 0 := by
      apply max_le
      · exact le_trans hSub (le_max_left _ _)
      · exact le_max_right _ _
    have hMaxMul :
        τ * max (a_min - input.a_L) 0 ≤
          τ * max (a_i - input.a_L) 0 :=
      mul_le_mul_of_nonneg_left hMax (le_of_lt τ_pos)
    unfold FeasibleZero SafeZero at *
    ring_nf at hSafe
    have hAccel : τ * a_min ≤ τ * a_i :=
      mul_le_mul_of_nonneg_left hBounds.1 (le_of_lt τ_pos)
    nlinarith

theorem negative_safe_implies_feasible
    (state : VehicleState) (hΔv : relSpeed state < 0)
    (hSafe : SafeSet state) :
    FeasibleNegative state := by
  have hBarrier : 0 ≤ ttcThwBarrier state := hSafe
  have hNumerator :
      0 ≤ -relSpeed state + κ * ttcThwBarrier state := by
    have hκBarrier : 0 ≤ κ * ttcThwBarrier state :=
      mul_nonneg (le_of_lt κ_pos) hBarrier
    linarith
  have hUpperNonnegative : 0 ≤ upperNegative state := by
    unfold upperNegative
    exact div_nonneg hNumerator (le_of_lt τ_pos)
  unfold FeasibleNegative
  exact le_trans a_min_le_zero hUpperNonnegative

theorem zero_safe_input_implies_feasible
    (state : VehicleState) (input : VehicleInput)
    (_hΔv : relSpeed state = 0)
    (hSafe : SafeSet state) (hInput : ValidInput input) :
    FeasibleZero state input := by
  have hSub : a_min - input.a_L ≤ 0 := by
    linarith [hInput.1]
  have hBarrier : 0 ≤ ttcThwBarrier state := hSafe
  have hκBarrier : 0 ≤ κ * ttcThwBarrier state :=
    mul_nonneg (le_of_lt κ_pos) hBarrier
  unfold FeasibleZero SafeZero
  rw [max_eq_right hSub]
  have hBrake : 0 ≤ -τ * a_min := by
    norm_num [τ, a_min]
  nlinarith

/-- Leader input representing the worst-case braking value used by the controller. -/
def hardBrakingInput : VehicleInput :=
  { a_L := a_min }

/-- Exact feasibility characterization under maximum leader braking. -/
theorem hard_braking_feasible_iff (state : VehicleState) :
    FeasiblePositive state hardBrakingInput ↔
      relSpeed state ≤ κ * ttcThwBarrier state - τ * a_min := by
  unfold FeasiblePositive upperPositive hardBrakingInput
  constructor
  · intro hF
    have hMul := (le_div_iff₀ two_τ_pos).1 hF
    nlinarith
  · intro hBound
    apply (le_div_iff₀ two_τ_pos).2
    nlinarith

theorem hard_braking_feasible_iff_numeric (state : VehicleState) :
    FeasiblePositive state hardBrakingInput ↔
      relSpeed state ≤ ttcThwBarrier state + 20 := by
  rw [hard_braking_feasible_iff]
  norm_num [τ, κ, a_min]

theorem hard_braking_infeasible_of_excess_closing_speed
    (state : VehicleState)
    (hExcess : ttcThwBarrier state + 20 < relSpeed state) :
    ¬ FeasiblePositive state hardBrakingInput := by
  rw [hard_braking_feasible_iff_numeric]
  linarith

/-! ## Controller implemented by `HighD_CBF.py` -/

/--
Worst-case upper bound used by the Python controller when `relSpeed ≥ 0`.
The implementation substitutes `a_min` for the leader acceleration.
-/
def robustUpperNonnegative (state : VehicleState) : ℝ :=
  (-relSpeed state + τ * a_min + κ * ttcThwBarrier state) / (2 * τ)

/-- Upper bound used by the Python controller when `relSpeed < 0`. -/
def implementedUpper (state : VehicleState) : ℝ :=
  if 0 ≤ relSpeed state then
    robustUpperNonnegative state
  else
    upperNegative state

/-- Feasibility of the scalar QP implemented in Python. -/
def ControllerFeasible (state : VehicleState) : Prop :=
  a_min ≤ implementedUpper state

/-- Linear inequality sent to the QP solver. -/
def ImplementedConstraint (state : VehicleState) (a_i : ℝ) : Prop :=
  a_i ≤ implementedUpper state

theorem controllerFeasible_iff_exists_bounded_control
    (state : VehicleState) :
    ControllerFeasible state ↔
      ∃ a_i : ℝ,
        PermittedAcceleration a_i ∧ ImplementedConstraint state a_i := by
  constructor
  · intro hF
    refine ⟨a_min, ⟨le_rfl, a_min_le_a_max⟩, ?_⟩
    simpa [ControllerFeasible, ImplementedConstraint] using hF
  · rintro ⟨a_i, hBounds, hConstraint⟩
    unfold ControllerFeasible ImplementedConstraint at *
    exact le_trans hBounds.1 hConstraint

/--
Closed form of the feasible scalar QP solution.  Since `a_min ≤ 0 ≤ a_max`,
this is the projection of zero onto `[a_min, min a_max implementedUpper]`.
-/
def qpControl (state : VehicleState) : ℝ :=
  max a_min (min 0 (implementedUpper state))

/--
The Python implementation returns zero when the solver supplies no value.
No CBF property is claimed for this fallback when the QP is infeasible.
-/
noncomputable def selectedControl (state : VehicleState) : ℝ :=
  by
    classical
    exact if ControllerFeasible state then qpControl state else 0

theorem qpControl_ge_a_min (state : VehicleState) :
    a_min ≤ qpControl state := by
  unfold qpControl
  exact le_max_left _ _

theorem qpControl_le_a_max (state : VehicleState) :
    qpControl state ≤ a_max := by
  unfold qpControl
  apply max_le
  · exact a_min_le_a_max
  · exact le_trans (min_le_left 0 (implementedUpper state)) zero_le_a_max

theorem qpControl_le_implementedUpper_of_feasible
    (state : VehicleState) (hF : ControllerFeasible state) :
    qpControl state ≤ implementedUpper state := by
  unfold ControllerFeasible at hF
  unfold qpControl
  apply max_le
  · exact hF
  · exact min_le_right 0 (implementedUpper state)

theorem qpControl_satisfies_implemented_constraint
    (state : VehicleState) (hF : ControllerFeasible state) :
    ImplementedConstraint state (qpControl state) := by
  exact qpControl_le_implementedUpper_of_feasible state hF

theorem selectedControl_eq_qp_of_feasible
    (state : VehicleState) (hF : ControllerFeasible state) :
    selectedControl state = qpControl state := by
  simp [selectedControl, hF]

theorem selectedControl_bounds (state : VehicleState) :
    PermittedAcceleration (selectedControl state) := by
  by_cases hF : ControllerFeasible state
  · rw [selectedControl_eq_qp_of_feasible state hF]
    exact ⟨qpControl_ge_a_min state, qpControl_le_a_max state⟩
  · norm_num [selectedControl, hF, PermittedAcceleration, a_min, a_max]

theorem selectedControl_satisfies_implemented_constraint
    (state : VehicleState) (hF : ControllerFeasible state) :
    ImplementedConstraint state (selectedControl state) := by
  rw [selectedControl_eq_qp_of_feasible state hF]
  exact qpControl_satisfies_implemented_constraint state hF

/-! ## Connection from the implemented robust constraint to the exact rate -/

lemma robust_constraint_implies_cbf_of_positive_relSpeed
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : 0 < relSpeed state)
    (hLeader : a_min ≤ input.a_L)
    (hRobust : a_i ≤ robustUpperNonnegative state) :
    PointwiseCBF state input a_i := by
  unfold robustUpperNonnegative at hRobust
  have hMul := (le_div_iff₀ two_τ_pos).1 hRobust
  have hLeaderMul : τ * a_min ≤ τ * input.a_L :=
    mul_le_mul_of_nonneg_left hLeader (le_of_lt τ_pos)
  unfold PointwiseCBF
  rw [barrierRate_of_positive_relSpeed state input a_i hΔv]
  nlinarith

lemma implemented_constraint_implies_cbf_of_negative_relSpeed
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state < 0)
    (hBound : a_i ≤ upperNegative state) :
    PointwiseCBF state input a_i := by
  exact safeNegative_implies_pointwise_cbf state input a_i hΔv hBound

/--
At zero relative speed, the robust Python constraint implies the exact
directional-rate inequality for every bounded leader acceleration.
-/
lemma robust_constraint_implies_cbf_of_zero_relSpeed
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hΔv : relSpeed state = 0)
    (hLeader : a_min ≤ input.a_L)
    (hFollower : a_min ≤ a_i)
    (hRobust : a_i ≤ robustUpperNonnegative state) :
    PointwiseCBF state input a_i := by
  have hMax : max (a_i - input.a_L) 0 ≤ a_i - a_min := by
    apply max_le
    · linarith
    · linarith
  have hMaxMul :
      τ * max (a_i - input.a_L) 0 ≤ τ * (a_i - a_min) :=
    mul_le_mul_of_nonneg_left hMax (le_of_lt τ_pos)
  unfold robustUpperNonnegative at hRobust
  have hMul := (le_div_iff₀ two_τ_pos).1 hRobust
  rw [hΔv] at hMul
  unfold PointwiseCBF
  rw [barrierRate_of_zero_relSpeed state input a_i hΔv]
  nlinarith

/-- The Python QP constraint implies the exact pointwise CBF inequality. -/
theorem implemented_constraint_implies_pointwise_cbf
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hInput : ValidInput input)
    (hBounds : PermittedAcceleration a_i)
    (hConstraint : ImplementedConstraint state a_i) :
    PointwiseCBF state input a_i := by
  have hLeader : a_min ≤ input.a_L := hInput.1
  have hFollower : a_min ≤ a_i := hBounds.1
  by_cases hPos : 0 < relSpeed state
  · have hRobust : a_i ≤ robustUpperNonnegative state := by
      unfold ImplementedConstraint implementedUpper at hConstraint
      simpa [le_of_lt hPos] using hConstraint
    exact robust_constraint_implies_cbf_of_positive_relSpeed
      state input a_i hPos hLeader hRobust
  · by_cases hNeg : relSpeed state < 0
    · have hNotNonnegative : ¬ 0 ≤ relSpeed state := not_le.mpr hNeg
      have hUpper : a_i ≤ upperNegative state := by
        unfold ImplementedConstraint implementedUpper at hConstraint
        simpa [hNotNonnegative] using hConstraint
      exact implemented_constraint_implies_cbf_of_negative_relSpeed
        state input a_i hNeg hUpper
    · have hZero : relSpeed state = 0 := by
        exact le_antisymm (le_of_not_gt hPos) (le_of_not_gt hNeg)
      have hRobust : a_i ≤ robustUpperNonnegative state := by
        unfold ImplementedConstraint implementedUpper at hConstraint
        simpa [hZero] using hConstraint
      exact robust_constraint_implies_cbf_of_zero_relSpeed
        state input a_i hZero hLeader hFollower hRobust

/--
The robust implementation constraint implies the exact branch-specific
acceleration condition under the stated input and follower bounds.
-/
theorem implemented_constraint_implies_safe_control
    (state : VehicleState) (input : VehicleInput) (a_i : ℝ)
    (hInput : ValidInput input)
    (hBounds : PermittedAcceleration a_i)
    (hConstraint : ImplementedConstraint state a_i) :
    SafeControl state input a_i := by
  have hCBF : PointwiseCBF state input a_i :=
    implemented_constraint_implies_pointwise_cbf
      state input a_i hInput hBounds hConstraint
  by_cases hPos : 0 < relSpeed state
  · have hBranch : SafePositive state input a_i :=
      (pointwise_cbf_iff_safePositive state input a_i hPos).1 hCBF
    simpa [SafeControl, hPos] using hBranch
  · by_cases hNeg : relSpeed state < 0
    · have hBranch : SafeNegative state a_i :=
        (pointwise_cbf_iff_safeNegative state input a_i hNeg).1 hCBF
      simpa [SafeControl, hPos, hNeg] using hBranch
    · have hZero : relSpeed state = 0 := by
        exact le_antisymm (le_of_not_gt hPos) (le_of_not_gt hNeg)
      have hBranch : SafeZero state input a_i :=
        (pointwise_cbf_iff_safeZero state input a_i hZero).1 hCBF
      simpa [SafeControl, hPos, hNeg] using hBranch

theorem selected_control_satisfies_cbf
    (state : VehicleState) (input : VehicleInput)
    (hInput : ValidInput input) (hF : ControllerFeasible state) :
    PointwiseCBF state input (selectedControl state) := by
  apply implemented_constraint_implies_pointwise_cbf
  · exact hInput
  · exact selectedControl_bounds state
  · exact selectedControl_satisfies_implemented_constraint state hF

theorem selected_control_satisfies_safe_control
    (state : VehicleState) (input : VehicleInput)
    (hInput : ValidInput input) (hF : ControllerFeasible state) :
    SafeControl state input (selectedControl state) := by
  apply implemented_constraint_implies_safe_control
  · exact hInput
  · exact selectedControl_bounds state
  · exact selectedControl_satisfies_implemented_constraint state hF

/--
Principal pointwise result.  It makes no trajectory-level invariance claim.
-/
theorem pointwise_safe_acceleration_exists
    (state : VehicleState) (input : VehicleInput)
    (_hState : ValidState state)
    (_hSafe : SafeSet state)
    (hInput : ValidInput input)
    (hF : ControllerFeasible state) :
    ∃ a_i : ℝ,
      PermittedAcceleration a_i ∧
      ImplementedConstraint state a_i ∧
      SafeControl state input a_i ∧
      PointwiseCBF state input a_i := by
  refine ⟨selectedControl state, selectedControl_bounds state, ?_, ?_, ?_⟩
  · exact selectedControl_satisfies_implemented_constraint state hF
  · exact selected_control_satisfies_safe_control state input hInput hF
  · exact selected_control_satisfies_cbf state input hInput hF

/-! ## Feasibility in the nonpositive-relative-speed regimes -/

theorem controller_feasible_of_negative_safe
    (state : VehicleState) (hΔv : relSpeed state < 0)
    (hSafe : SafeSet state) :
    ControllerFeasible state := by
  have hBarrier : 0 ≤ ttcThwBarrier state := hSafe
  have hNumerator :
      0 ≤ -relSpeed state + κ * ttcThwBarrier state := by
    have hκBarrier : 0 ≤ κ * ttcThwBarrier state :=
      mul_nonneg (le_of_lt κ_pos) hBarrier
    linarith
  have hUpperNonnegative : 0 ≤ upperNegative state := by
    unfold upperNegative
    exact div_nonneg hNumerator (le_of_lt τ_pos)
  unfold ControllerFeasible implementedUpper
  rw [if_neg (not_le.mpr hΔv)]
  exact le_trans a_min_le_zero hUpperNonnegative

theorem controller_feasible_of_zero_safe
    (state : VehicleState) (hΔv : relSpeed state = 0)
    (hSafe : SafeSet state) :
    ControllerFeasible state := by
  have hBarrier : 0 ≤ ttcThwBarrier state := hSafe
  unfold ControllerFeasible implementedUpper
  rw [if_pos (by simp [hΔv])]
  unfold robustUpperNonnegative
  rw [hΔv]
  apply (le_div_iff₀ two_τ_pos).2
  norm_num [τ, κ, a_min] at hBarrier ⊢
  linarith

theorem controller_feasible_of_nonpositive_safe
    (state : VehicleState) (hΔv : relSpeed state ≤ 0)
    (hSafe : SafeSet state) :
    ControllerFeasible state := by
  rcases hΔv.lt_or_eq with hNeg | hZero
  · exact controller_feasible_of_negative_safe state hNeg hSafe
  · exact controller_feasible_of_zero_safe state hZero hSafe

/-! ## Concrete validation cases -/

namespace Validation

def nonClosingState : VehicleState :=
  { x_L := 40, v_L := 12, x_i := 10, v_i := 10, L := 5 }

def closingFeasibleState : VehicleState :=
  { x_L := 55, v_L := 8, x_i := 10, v_i := 10, L := 5 }

def closingInfeasibleState : VehicleState :=
  { x_L := 195, v_L := 6, x_i := 10, v_i := 39, L := 5 }

def zeroRelativeSpeedState : VehicleState :=
  { x_L := 40, v_L := 10, x_i := 10, v_i := 10, L := 5 }

def hardBrakingBoundaryState : VehicleState :=
  { x_L := 140, v_L := 10, x_i := 10, v_i := 30, L := 5 }

def hardBrakingBeyondBoundaryState : VehicleState :=
  { x_L := 145, v_L := 10, x_i := 10, v_i := 31, L := 5 }

example : ValidState nonClosingState := by
  norm_num [ValidState, nonClosingState, spacing]

example : relSpeed nonClosingState < 0 := by
  norm_num [relSpeed, nonClosingState]

example : SafeSet nonClosingState := by
  norm_num [SafeSet, ttcThwBarrier, spacing, relSpeed, nonClosingState, τ]

example : ControllerFeasible nonClosingState := by
  norm_num [ControllerFeasible, implementedUpper, upperNegative,
    robustUpperNonnegative, ttcThwBarrier, spacing, relSpeed,
    nonClosingState, τ, κ, a_min]

example : ValidState closingFeasibleState := by
  norm_num [ValidState, closingFeasibleState, spacing]

example : 0 < relSpeed closingFeasibleState := by
  norm_num [relSpeed, closingFeasibleState]

example : SafeSet closingFeasibleState := by
  norm_num [SafeSet, ttcThwBarrier, spacing, relSpeed, closingFeasibleState, τ]

example : ControllerFeasible closingFeasibleState := by
  norm_num [ControllerFeasible, implementedUpper, robustUpperNonnegative,
    upperNegative, ttcThwBarrier, spacing, relSpeed,
    closingFeasibleState, τ, κ, a_min]

example : ValidState closingInfeasibleState := by
  norm_num [ValidState, closingInfeasibleState, spacing]

example : 0 < relSpeed closingInfeasibleState := by
  norm_num [relSpeed, closingInfeasibleState]

example : SafeSet closingInfeasibleState := by
  norm_num [SafeSet, ttcThwBarrier, spacing, relSpeed,
    closingInfeasibleState, τ]

example : ¬ ControllerFeasible closingInfeasibleState := by
  norm_num [ControllerFeasible, implementedUpper, robustUpperNonnegative,
    upperNegative, ttcThwBarrier, spacing, relSpeed,
    closingInfeasibleState, τ, κ, a_min]

example : relSpeed zeroRelativeSpeedState = 0 := by
  norm_num [relSpeed, zeroRelativeSpeedState]

example : ValidState zeroRelativeSpeedState := by
  norm_num [ValidState, zeroRelativeSpeedState, spacing]

example : hardBrakingInput.a_L < (0 : ℝ) := by
  norm_num [hardBrakingInput, a_min]

example :
    barrierRate zeroRelativeSpeedState hardBrakingInput 0 +
      κ * ttcThwBarrier zeroRelativeSpeedState = -20 := by
  norm_num [barrierRate, hardBrakingInput, zeroRelativeSpeedState,
    ttcThwBarrier, spacing, relSpeed, τ, κ, a_min]

example : ¬ SafeZero zeroRelativeSpeedState hardBrakingInput 0 := by
  norm_num [SafeZero, hardBrakingInput, zeroRelativeSpeedState,
    ttcThwBarrier, spacing, relSpeed, τ, κ, a_min]

example : ValidState hardBrakingBoundaryState := by
  norm_num [ValidState, hardBrakingBoundaryState, spacing]

example : ttcThwBarrier hardBrakingBoundaryState = 0 := by
  norm_num [ttcThwBarrier, hardBrakingBoundaryState, spacing, relSpeed, τ]

example :
    relSpeed hardBrakingBoundaryState =
      ttcThwBarrier hardBrakingBoundaryState + 20 := by
  norm_num [ttcThwBarrier, hardBrakingBoundaryState, spacing, relSpeed, τ]

example : FeasiblePositive hardBrakingBoundaryState hardBrakingInput := by
  norm_num [FeasiblePositive, upperPositive, hardBrakingInput,
    ttcThwBarrier, hardBrakingBoundaryState, spacing, relSpeed,
    τ, κ, a_min]

example : ControllerFeasible hardBrakingBoundaryState := by
  norm_num [ControllerFeasible, implementedUpper, robustUpperNonnegative,
    upperNegative, ttcThwBarrier, spacing, relSpeed,
    hardBrakingBoundaryState, τ, κ, a_min]

example : ValidState hardBrakingBeyondBoundaryState := by
  norm_num [ValidState, hardBrakingBeyondBoundaryState, spacing]

example : ttcThwBarrier hardBrakingBeyondBoundaryState = 0 := by
  norm_num [ttcThwBarrier, hardBrakingBeyondBoundaryState, spacing, relSpeed, τ]

example :
    ttcThwBarrier hardBrakingBeyondBoundaryState + 20 <
      relSpeed hardBrakingBeyondBoundaryState := by
  norm_num [ttcThwBarrier, hardBrakingBeyondBoundaryState,
    spacing, relSpeed, τ]

example : ¬ FeasiblePositive hardBrakingBeyondBoundaryState hardBrakingInput := by
  apply hard_braking_infeasible_of_excess_closing_speed
  norm_num [ttcThwBarrier, hardBrakingBeyondBoundaryState,
    spacing, relSpeed, τ]

example : ¬ ControllerFeasible hardBrakingBeyondBoundaryState := by
  norm_num [ControllerFeasible, implementedUpper, robustUpperNonnegative,
    upperNegative, ttcThwBarrier, spacing, relSpeed,
    hardBrakingBeyondBoundaryState, τ, κ, a_min]

/-- The combined barrier is strictly stronger than the two individual gap inequalities. -/
def individualGapCounterexample : VehicleState :=
  { x_L := 40, v_L := 8, x_i := 10, v_i := 10, L := 5 }

example :
    ValidState individualGapCounterexample ∧
    0 < relSpeed individualGapCounterexample ∧
    τ * individualGapCounterexample.v_i ≤ spacing individualGapCounterexample ∧
    τ * relSpeed individualGapCounterexample ≤ spacing individualGapCounterexample ∧
    ¬ SafeSet individualGapCounterexample := by
  norm_num [ValidState, individualGapCounterexample, spacing, relSpeed,
    SafeSet, ttcThwBarrier, τ]

end Validation

end
end CBFProject
