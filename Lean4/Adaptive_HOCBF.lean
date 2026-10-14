import Mathlib

namespace DBCCProject
set_option linter.style.longLine false
noncomputable section


/-! ## 1. Core Structures -/

structure VehicleState where
  x_L : ℝ          -- Position of the leading vehicle
  v_L : ℝ          -- Velocity of the leading vehicle
  x_i : ℝ          -- Position of the following vehicle
  v_i : ℝ          -- Velocity of the following vehicle

structure VehicleInput where
  a_L : ℝ          -- Acceleration of the leading vehicle (external disturbance)
  a_i : ℝ          -- Acceleration command of the following vehicle (control input)


structure DeltaParams where
  delta_base : ℝ
  delta_max : ℝ
  M_rate : ℝ
  M2     : ℝ   -- |¨δ|
  M_vel  : ℝ
  h_base_pos : delta_base > 0
  h_max_ge : delta_max ≥ delta_base
  h_rate_pos : M_rate > 0
  h_M2_pos   : M2 > 0
  h_Mvel_pos : M_vel > 0

variable (params : DeltaParams)


/-! ## 2. Physical Bounds -/

def a_min : ℝ := -8
def a_max : ℝ := 3

def ValidInput (input : VehicleInput) : Prop :=
  a_min ≤ input.a_L ∧ input.a_L ≤ a_max ∧
  a_min ≤ input.a_i ∧ input.a_i ≤ a_max

def ValidState (state : VehicleState) : Prop :=
  state.x_L > state.x_i ∧
  0 < state.v_L ∧ state.v_L < 100 ∧
  0 < state.v_i ∧ state.v_i < 100

/-! ## 4. Spacing Error (Continuous Time) -/

def spacing_error (state : VehicleState) (d_des : ℝ) : ℝ :=
  (state.x_L - state.x_i) - d_des

/-! ## 5. Time-Varying Tolerance δ(t) -/

-- Class-K function parameters (design gains)
def alpha1 : ℝ := 1.0
def alpha2 : ℝ := 1.0

/-! ## 6. Barrier Function (Time-Varying) -/

def barrier (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  let e := spacing_error state d_des
  (δ t)^2 - e^2

def SafeSet (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : Prop :=
  barrier δ state d_des t ≥ 0

/-! ## 7. Error Dynamics (Continuous Time) -/

-- ė = v_L - v_i
def error_derivative (state : VehicleState) : ℝ :=
  state.v_L - state.v_i

-- ë = a_L - a_i   (control input appears here)
def error_second_derivative (input : VehicleInput) : ℝ :=
  input.a_L - input.a_i

/-! ## 8. Derivatives of the Barrier Function (Continuous Time) -/

-- First total derivative of the barrier: ḣ = ∂h/∂t + (∂h/∂e) * ė
def barrier_derivative (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  let e := spacing_error state d_des
  -- ∂h/∂t = 2 * δ(t) * δ̇(t)
  let dh_dt := 2 * (δ t) * (deriv δ t)
  -- ∂h/∂e = -2e
  let dh_de := -2 * e
  dh_dt + dh_de * error_derivative state

-- Second total derivative of the barrier: ḧ
def barrier_second_derivative (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) (t : ℝ) : ℝ :=
  let e := spacing_error state d_des
  let de := error_derivative state
  let dde := error_second_derivative input
  -- Correct second derivative of h = δ(t)² - e²
  2 * (deriv δ t)^2
    + 2 * (δ t) * (deriv (deriv δ) t)
    - 2 * de^2
    - 2 * e * dde

/-! ## 9. Higher-Order CBF Auxiliary Functions (Time-Varying) -/

/-- ψ₀ = h (the barrier function itself) -/
def psi0 (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  barrier δ state d_des t

/-- ψ₁ = ḣ + α₁(h) · h -/
def psi1 (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  let h_val := psi0 δ state d_des t
  let h_dot := barrier_derivative δ state d_des t
  h_dot + alpha1 * h_val

/-! ## 10. Higher-Order Time-Varying CBF Condition -/

-- Derivative of ψ₁ (this is the most complex expression)
def psi1_derivative (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) (t : ℝ) : ℝ :=
  let h_dot := barrier_derivative δ state d_des t
  let h_ddot := barrier_second_derivative δ state d_des input t
  -- Derivative of (ḣ + α₁ · h)
  h_ddot + alpha1 * h_dot

-- Final Higher-Order Time-Varying CBF Condition
-- ḣ₁ + α₂(ψ₁) · ψ₁ ≥ 0
def HOCBF_Condition (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) (t : ℝ) : Prop :=
  let psi1_val := psi1 δ state d_des t
  let psi1_dot := psi1_derivative δ state d_des input t
  psi1_dot + alpha2 * psi1_val ≥ 0

/-! ## 11. Exact Control Lower Bound (Derived from CBF Condition) -/

lemma a_min_le_zero :
    a_min ≤ 0 := by
  norm_num [a_min]

lemma zero_le_a_max :
    (0 : ℝ) ≤ a_max := by
  norm_num [a_max]

def hocbf_num (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  let e  := spacing_error state d_des
  let de := error_derivative state
  let δ' := deriv δ t
  let δ'' := deriv (deriv δ) t
  let h  := barrier δ state d_des t
  δ'^2 + (δ t)*δ'' - de^2
  + (alpha1 + alpha2)*(δ t)*δ'
  - (alpha1 + alpha2)*e*de
  + (alpha1 * alpha2 / 2) * h

/-- Exact lower bound on a_i when e > 0.
    Derived from  ψ̇₁ + α₂ ψ₁ ≥ 0  after collecting the coefficient of a_i. -/
def control_lower_bound (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ)
    (input : VehicleInput) (t : ℝ) : ℝ :=
  input.a_L - (hocbf_num δ state d_des t) / (spacing_error state d_des)
/-! ## 12. Exact Control Upper Bound (for e < 0) -/

/-- Exact upper bound on a_i when e < 0. -/
def control_upper_bound (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ)
    (input : VehicleInput) (t : ℝ) : ℝ :=
  input.a_L - (hocbf_num δ state d_des t) / (spacing_error state d_des)

lemma min_upper_nominal_valid
    (upper nominal_a_i : ℝ)
    (h_upper_lower :
      a_min ≤ upper)
    (h_nominal_lower :
      a_min ≤ nominal_a_i)
    (h_upper_max :
      upper ≤ a_max) :
    a_min ≤ min upper nominal_a_i ∧
    min upper nominal_a_i ≤ a_max := by
  constructor
  · exact le_min h_upper_lower h_nominal_lower
  · exact le_trans (min_le_left upper nominal_a_i) h_upper_max


lemma max_lower_nominal_valid
    (lower nominal_a_i : ℝ)
    (h_lower_min :
      a_min ≤ lower)
    (h_lower_max :
      lower ≤ a_max)
    (h_nominal_max :
      nominal_a_i ≤ a_max) :
    a_min ≤ max lower nominal_a_i ∧
    max lower nominal_a_i ≤ a_max := by
  constructor
  · exact le_trans h_lower_min (le_max_left lower nominal_a_i)
  · exact max_le h_lower_max h_nominal_max


/-- Any acceleration command that respects the CBF-derived bounds -/
def safe_control_input
    (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ)
    (a_L : ℝ) (nominal_a_i : ℝ) : VehicleInput :=
  let lower := control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t
  let upper := control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t
  let e := spacing_error state d_des
  if e < 0 then
    { a_L := a_L, a_i := min upper nominal_a_i }
  else if e > 0 then
    { a_L := a_L, a_i := max lower nominal_a_i }
  else
    { a_L := a_L, a_i := nominal_a_i }

lemma safe_control_input_valid
    (δ : ℝ → ℝ) (state : VehicleState) (d_des : ℝ) (t : ℝ)
    (a_L : ℝ) (nominal_a_i : ℝ)
    (h_aL : a_min ≤ a_L ∧ a_L ≤ a_max)
    (h_nominal : a_min ≤ nominal_a_i ∧ nominal_a_i ≤ a_max)
    (h_upper_lower :
      a_min ≤ control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t)
    (h_upper_max :
      control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t ≤ a_max)
    (h_lower_min :
      a_min ≤ control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t)
    (h_lower_max :
      control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } t ≤ a_max) :
    ValidInput (safe_control_input δ state d_des t a_L nominal_a_i) := by
  unfold ValidInput
  unfold safe_control_input
  dsimp
  by_cases h_neg :
      spacing_error state d_des < 0
  · -- e < 0 branch
    simp only [h_neg, if_pos]
    have h :=
      min_upper_nominal_valid
        (control_upper_bound δ state d_des
          {a_L := a_L, a_i := nominal_a_i} t)
        nominal_a_i
        h_upper_lower
        h_nominal.1
        h_upper_max
    constructor
    · exact h_aL.1
    constructor
    · exact h_aL.2
    exact h
  · by_cases h_pos :
        spacing_error state d_des > 0
    · -- e > 0 branch
      simp only [h_neg, h_pos]
      have h :=
        max_lower_nominal_valid
          (control_lower_bound δ state d_des
            {a_L := a_L, a_i := nominal_a_i} t)
          nominal_a_i
          h_lower_min
          h_lower_max
          h_nominal.2
      constructor
      · exact h_aL.1
      constructor
      · exact h_aL.2
      exact h
    · -- e = 0 branch
      simp only [h_neg, h_pos]
      constructor
      · exact h_aL.1
      constructor
      · exact h_aL.2
      constructor
      · exact h_nominal.1
      · exact h_nominal.2


/-- This is the key predicate — it says "δ behaves as the Python code promises" -/
def delta_implements_adaptive_law (params : DeltaParams) (δ : ℝ → ℝ) : Prop :=
  (∀ t, params.delta_base ≤ δ t ∧ δ t ≤ params.delta_max) ∧
  (∀ t, |deriv δ t| ≤ params.M_rate) ∧
  (∀ t, |deriv (deriv δ) t| ≤ params.M2) ∧
  LocallyLipschitz δ

lemma delta_adaptive_regular {params : DeltaParams} {δ : ℝ → ℝ}
    (h : delta_implements_adaptive_law params δ) :
    (∀ t, δ t > 0) ∧
    LocallyLipschitz δ ∧
    (∀ t, |deriv δ t| ≤ params.M_rate) ∧
    (∀ t, |deriv (deriv δ) t| ≤ params.M2) := by
  obtain ⟨h_bounds, h_deriv, h_dderiv, h_lip⟩ := h
  constructor
  · intro t
    have hb := h_bounds t
    linarith [hb.1, params.h_base_pos]
  · exact ⟨h_lip, h_deriv, h_dderiv⟩


/-! ## 15. Equivalence between CBF Condition and Control Bound -/

theorem control_lower_bound_implies_HOCBF
    (δ : ℝ → ℝ)
    (state : VehicleState)
    (d_des : ℝ)
    (input : VehicleInput)
    (t : ℝ)
    (h_e : spacing_error state d_des > 0)
    (h_bound : input.a_i ≥ control_lower_bound δ state d_des input t) :
    HOCBF_Condition δ state d_des input t := by
  unfold HOCBF_Condition psi1 psi1_derivative
  unfold barrier_derivative barrier_second_derivative psi0 barrier
  simp only [spacing_error, error_derivative, error_second_derivative]
  set e   := spacing_error state d_des with he
  set de  := error_derivative state with hde
  set num := hocbf_num δ state d_des t with hnum
  -- ------------------------------------------------------------------
  -- 1. Rearrange the hypothesis into the convenient form
  --    a_i - a_L ≥ - num / e
  -- ------------------------------------------------------------------
  have h1 : input.a_i - input.a_L ≥ - num / e := by
    -- h_bound : a_i ≥ a_L - num/e
    -- which is the same as a_i - a_L ≥ - num/e
    -- after moving a_L to the left-hand side.
    have : input.a_i ≥ input.a_L - num / e := by
      simpa [control_lower_bound, he, hnum] using h_bound
    ring_nf at this
    simp [he, hnum]
    ring_nf
    linarith [this]
  -- ------------------------------------------------------------------
  -- 2. Multiply by the positive quantity e
  -- ------------------------------------------------------------------
  have he_pos : 0 < e := by simpa [he] using h_e
  have h2 : e * (input.a_i - input.a_L) ≥ - num := by
    -- multiply both sides of h1 by e > 0
    have := mul_le_mul_of_nonneg_left (ge_iff_le.mp h1) (le_of_lt he_pos)
    -- right-hand side: (-num/e)*e = -num
    field_simp [ne_of_gt he_pos] at this
    exact this
  -- ------------------------------------------------------------------
  -- 3. Multiply by 2 and obtain the target non-negativity statement
  -- ------------------------------------------------------------------
  have h_target : 0 ≤ 2 * e * (input.a_i - input.a_L) + 2 * num := by
    have h3 : 2 * e * (input.a_i - input.a_L) ≥ -2 * num := by
      linarith [h2]
    linarith [h3]
  -- ------------------------------------------------------------------
  -- 4. Identify the expanded HOCBF expression with the quantity above
  -- ------------------------------------------------------------------
  convert h_target using 1
  · simp only [num, he]
    unfold spacing_error hocbf_num
    simp only [spacing_error, error_derivative, barrier]
    ring


theorem control_upper_bound_implies_HOCBF
    (δ : ℝ → ℝ)
    (state : VehicleState)
    (d_des : ℝ)
    (input : VehicleInput)
    (t : ℝ)
    (h_e : spacing_error state d_des < 0)
    (h_bound : input.a_i ≤ control_upper_bound δ state d_des input t) :
    HOCBF_Condition δ state d_des input t := by
  unfold HOCBF_Condition psi1 psi1_derivative
  unfold barrier_derivative barrier_second_derivative psi0 barrier
  simp only [spacing_error, error_derivative, error_second_derivative]
  set e   := spacing_error state d_des with he
  set de  := error_derivative state with hde
  set num := hocbf_num δ state d_des t with hnum
  -- ------------------------------------------------------------------
  -- 1. Rearrange the hypothesis into the convenient form
  --    a_i - a_L ≤ - num / e
  -- ------------------------------------------------------------------
  have h1 : input.a_i - input.a_L ≤ - num / e := by
    -- h_bound : a_i ≤ a_L - num/e
    -- which is the same as a_i - a_L ≤ - num/e
    have : input.a_i ≤ input.a_L - num / e := by
      simpa [control_upper_bound, he, hnum] using h_bound
    ring_nf at this
    simp [he, hnum]
    ring_nf
    linarith [this]
  -- ------------------------------------------------------------------
  -- 2. Multiply by the negative quantity e (inequality reverses)
  -- ------------------------------------------------------------------
  have he_neg : e < 0 := by simpa [he] using h_e
  have h2 : e * (input.a_i - input.a_L) ≥ - num := by
    -- multiply both sides of h1 by e < 0  → inequality reverses
    have := mul_le_mul_of_nonpos_left h1 (le_of_lt he_neg)
    -- right-hand side: (-num/e)*e = -num
    field_simp [ne_of_lt he_neg] at this
    simpa [ne_of_lt he_neg] using this
  -- ------------------------------------------------------------------
  -- 3. Multiply by 2 and obtain the target non-negativity statement
  -- ------------------------------------------------------------------
  have h_target : 0 ≤ 2 * e * (input.a_i - input.a_L) + 2 * num := by
    have h3 : 2 * e * (input.a_i - input.a_L) ≥ -2 * num := by
      linarith [h2]
    linarith [h3]
  -- ------------------------------------------------------------------
  -- 4. Identify the expanded HOCBF expression with the quantity above
  -- ------------------------------------------------------------------
  convert h_target using 1
  · simp only [num, he]
    unfold spacing_error hocbf_num
    simp only [spacing_error, error_derivative, barrier]
    ring


/-! ## 16. Closed-Loop Dynamics and Global Safety -/
/-- Initial conditions for HOCBF safety -/
def initial_hocbf_conditions (δ : ℝ → ℝ) (state0 : VehicleState) (d_des : ℝ) : Prop :=
  psi0 δ state0 d_des 0 ≥ 0 ∧
  psi1 δ state0 d_des 0 ≥ 0

/-! ## 16. Global Safety -/

def ZeroErrorVelocityBound (params : DeltaParams) (state : VehicleState) : Prop :=
  |state.v_L - state.v_i| ≤ params.M_vel

/-- Gain condition guaranteeing HOCBF satisfaction when the spacing error is zero. -/
def ZeroErrorGainCondition (params : DeltaParams) : Prop :=
  - 2 * params.delta_max * params.M2
  - 2 * (alpha1 + alpha2) * params.delta_max * params.M_rate
  + alpha1 * alpha2 * params.delta_base^2
  ≥
  2 * params.M_vel^2

theorem zero_error_implies_HOCBF
    (δ : ℝ → ℝ)
    (params : DeltaParams)
    (h_adapt : delta_implements_adaptive_law params δ)
    (state : VehicleState)
    (d_des : ℝ)
    (input : VehicleInput)
    (t : ℝ)
    (h_e_zero : spacing_error state d_des = 0)
    (h_vel : ZeroErrorVelocityBound params state)
    (h_gain : ZeroErrorGainCondition params) :
    HOCBF_Condition δ state d_des input t := by
  have h_reg := delta_adaptive_regular h_adapt
  have h_e_zero' :
        state.x_L - state.x_i - d_des = 0 := by
      simpa [spacing_error] using h_e_zero
  unfold HOCBF_Condition
  unfold psi1 psi1_derivative
  unfold barrier_derivative barrier_second_derivative psi0
  simp only [neg_mul, ge_iff_le]
  have h_delta_pos : δ t > 0 :=
    h_reg.1 t
  have h_delta_lower :
      params.delta_base ≤ δ t :=
    (h_adapt.1 t).1
  have h_delta_upper :
      δ t ≤ params.delta_max :=
    (h_adapt.1 t).2
  have h_d1 :
      |deriv δ t| ≤ params.M_rate :=
    h_reg.2.2.1 t
  have h_d2 :
      |deriv (deriv δ) t| ≤ params.M2 :=
    h_reg.2.2.2 t
  have h_vel_sq :
      (state.v_L - state.v_i)^2 ≤ params.M_vel^2 := by
    have hvel :
        -params.M_vel ≤ state.v_L - state.v_i ∧
        state.v_L - state.v_i ≤ params.M_vel :=
      abs_le.mp h_vel
    have hM : 0 ≤ params.M_vel := by
      exact le_of_lt params.h_Mvel_pos
    nlinarith
  have h_cross_ddot :
      -2 * params.delta_max * params.M2
      ≤
      2 * δ t * deriv (deriv δ) t := by
    have hδ0 : 0 ≤ δ t :=
      le_of_lt h_delta_pos
    have hddot :
        -params.M2 ≤ deriv (deriv δ) t :=
      (abs_le.mp h_d2).1
    have h1 :
        δ t * (-params.M2)
        ≤
        δ t * deriv (deriv δ) t :=
      mul_le_mul_of_nonneg_left hddot hδ0
    have h2 :
        -(params.delta_max * params.M2)
        ≤
        δ t * (-params.M2) := by
      have hM2 :
          0 ≤ params.M2 :=
        le_of_lt params.h_M2_pos
      nlinarith [h_delta_upper]
    have h3 :
        -(params.delta_max * params.M2)
        ≤
        δ t * deriv (deriv δ) t :=
      le_trans h2 h1
    nlinarith
  have h_cross_dot :
      -2 * (alpha1 + alpha2)
        * params.delta_max
        * params.M_rate
      ≤
      2 * (alpha1 + alpha2)
        * δ t * deriv δ t := by
    have hδ0 :
        0 ≤ δ t :=
      le_of_lt h_delta_pos
    have hd :
        -params.M_rate ≤ deriv δ t :=
      (abs_le.mp h_d1).1
    have h1 :
        δ t * (-params.M_rate)
        ≤
        δ t * deriv δ t :=
      mul_le_mul_of_nonneg_left hd hδ0
    have h2 :
        -(params.delta_max * params.M_rate)
        ≤
        δ t * (-params.M_rate) := by
      have hM :
          0 ≤ params.M_rate :=
        le_of_lt params.h_rate_pos
      nlinarith [h_delta_upper]
    have h3 :
        -(params.delta_max * params.M_rate)
        ≤
        δ t * deriv δ t :=
      le_trans h2 h1
    have hα :
        0 ≤ 2 * (alpha1 + alpha2) := by
      norm_num [alpha1, alpha2]
    have h4 :=
      mul_le_mul_of_nonneg_left h3 hα
    simpa [mul_assoc,
           mul_left_comm,
           mul_comm] using h4
  have h_alpha_term :
      alpha1 * alpha2 * params.delta_base^2
      ≤
      alpha1 * alpha2 * (δ t)^2 := by
    have hα :
        0 ≤ alpha1 * alpha2 := by
      norm_num [alpha1, alpha2]
    have hsq :
        params.delta_base^2 ≤ (δ t)^2 := by
      nlinarith [h_delta_lower,
                 params.h_base_pos]
    exact mul_le_mul_of_nonneg_left hsq hα
  have h_main :
      0 ≤
      2 * (deriv δ t)^2
      + 2 * δ t * deriv (deriv δ) t
      - 2 * (state.v_L - state.v_i)^2
      + 2 * (alpha1 + alpha2) * δ t * deriv δ t
      + alpha1 * alpha2 * (δ t)^2 := by
    have h_gain' :
      -2 * params.delta_max * params.M2
      -2 * (alpha1 + alpha2) * params.delta_max * params.M_rate
      + alpha1 * alpha2 * params.delta_base^2
      ≥
      2 * params.M_vel^2 := by
      exact h_gain
    nlinarith [
    sq_nonneg (deriv δ t),
    h_cross_ddot,
    h_cross_dot,
    h_alpha_term,
    h_vel_sq,
    h_gain']
  unfold barrier
  simp [h_e_zero, error_derivative]
  ring_nf at *
  exact h_main

/-- The closed-loop controller always satisfies the HOCBF condition. -/
theorem closed_loop_satisfies_hocbf
    (δ : ℝ → ℝ) (params : DeltaParams)
    (h_adapt : delta_implements_adaptive_law params δ)
    (d_des : ℝ) (t : ℝ) (state : VehicleState)
    (a_L : ℝ) (nominal_a_i : ℝ)
    (h_vel : ZeroErrorVelocityBound params state)
    (h_gain : ZeroErrorGainCondition params) :
    HOCBF_Condition δ state d_des
      (safe_control_input δ state d_des t a_L nominal_a_i) t := by
  by_cases h_neg : spacing_error state d_des < 0
  · -- e < 0  →  upper bound
    have h_safe :
        safe_control_input δ state d_des t a_L nominal_a_i =
          { a_L := a_L,
            a_i := min (control_upper_bound δ state d_des
                          { a_L := a_L, a_i := nominal_a_i } t)
                      nominal_a_i } := by
      unfold safe_control_input
      simp only [h_neg, if_pos]
    rw [h_safe]
    apply control_upper_bound_implies_HOCBF
    · exact h_neg
    · -- a_i = min(upper, nominal) ≤ upper
      exact min_le_left _ _
  · by_cases h_pos : spacing_error state d_des > 0
    · -- e > 0  →  lower bound
      have h_safe :
          safe_control_input δ state d_des t a_L nominal_a_i =
            { a_L := a_L,
              a_i := max (control_lower_bound δ state d_des
                            { a_L := a_L, a_i := nominal_a_i } t)
                        nominal_a_i } := by
        unfold safe_control_input
        simp [h_neg, h_pos]
      rw [h_safe]
      apply control_lower_bound_implies_HOCBF
      · exact h_pos
      · -- a_i = max(lower, nominal) ≥ lower
        exact le_max_left _ _
    · -- e = 0
      have h_zero : spacing_error state d_des = 0 := by
        apply le_antisymm <;> linarith
      have h_safe :
          safe_control_input δ state d_des t a_L nominal_a_i =
            { a_L := a_L, a_i := nominal_a_i } := by
        unfold safe_control_input
        simp [h_zero, lt_self_iff_false]
      rw [h_safe]
      exact zero_error_implies_HOCBF δ params h_adapt state d_des
        { a_L := a_L, a_i := nominal_a_i } t h_zero h_vel h_gain


def is_closed_loop_trajectory
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ) : Prop :=
  ∀ t,
    HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t ∧
    HasDerivAt (fun τ => (traj τ).v_L) (a_L t) t ∧
    HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t ∧
    HasDerivAt (fun τ => (traj τ).v_i)
      (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i t ∧
    (a_min ≤ a_L t ∧ a_L t ≤ a_max)

lemma closed_loop_hocbf_along_trajectory
    (δ : ℝ → ℝ) (params : DeltaParams)
    (h_adapt : delta_implements_adaptive_law params δ)
    (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_vel : ∀ t ≥ 0, ZeroErrorVelocityBound params (traj t))
    (h_gain : ZeroErrorGainCondition params) :
    ∀ t ≥ 0,
      HOCBF_Condition δ (traj t) d_des
        (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t := by
  intro t ht
  exact closed_loop_satisfies_hocbf
    δ
    params
    h_adapt
    d_des
    t
    (traj t)
    (a_L t)
    nominal_a_i
    (h_vel t ht)
    h_gain


/-! ## Chain-rule lemmas along closed-loop trajectories -/

lemma barrier_deriv_along_traj
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) :
    ∀ t, HasDerivAt (fun τ => barrier δ (traj τ) d_des τ)
          (barrier_derivative δ (traj t) d_des t) t := by
  intro t
  -- derivative of δ
  have hδ : HasDerivAt δ (deriv δ t) t := (h_δ t).hasDerivAt
  -- kinematic derivatives from the closed-loop hypothesis
  have hxL : HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t := (h_traj t).1
  have hxi : HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t := (h_traj t).2.2.1
  -- ------------------------------------------------------------------
  -- δ · δ
  -- ------------------------------------------------------------------
  have hδ_sq : HasDerivAt (fun τ => δ τ * δ τ) (2 * δ t * deriv δ t) t := by
    have h_prod : HasDerivAt (fun τ => δ τ * δ τ)
        (deriv δ t * δ t + δ t * deriv δ t) t := hδ.mul hδ
    exact h_prod.congr_deriv (by ring)
  -- ------------------------------------------------------------------
  -- e = x_L - x_i - d_des
  -- ------------------------------------------------------------------
  have he : HasDerivAt
      (fun τ => (traj τ).x_L - (traj τ).x_i - d_des)
      ((traj t).v_L - (traj t).v_i) t := by
    have h1 : HasDerivAt (fun τ => (traj τ).x_L - (traj τ).x_i)
        ((traj t).v_L - (traj t).v_i) t := hxL.sub hxi
    have h2 : HasDerivAt (fun _ => (d_des : ℝ)) 0 t := hasDerivAt_const t d_des
    have h3 : HasDerivAt
        (fun τ => (traj τ).x_L - (traj τ).x_i - d_des)
        ((traj t).v_L - (traj t).v_i - 0) t := h1.sub h2
    convert h3 using 1
    · ring
  -- ------------------------------------------------------------------
  -- e · e
  -- ------------------------------------------------------------------
  have he_sq : HasDerivAt
      (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                * ((traj τ).x_L - (traj τ).x_i - d_des))
      (2 * ((traj t).x_L - (traj t).x_i - d_des)
          * ((traj t).v_L - (traj t).v_i)) t := by
    have h_prod : HasDerivAt
        (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                  * ((traj τ).x_L - (traj τ).x_i - d_des))
        (((traj t).v_L - (traj t).v_i)
            * ((traj t).x_L - (traj t).x_i - d_des)
          + ((traj t).x_L - (traj t).x_i - d_des)
            * ((traj t).v_L - (traj t).v_i)) t := he.mul he
    exact h_prod.congr_deriv (by ring)
  -- ------------------------------------------------------------------
  -- δ·δ - e·e
  -- ------------------------------------------------------------------
  have h_barrier : HasDerivAt
      (fun τ => δ τ * δ τ
                - ((traj τ).x_L - (traj τ).x_i - d_des)
                    * ((traj τ).x_L - (traj τ).x_i - d_des))
      (2 * δ t * deriv δ t
        - 2 * ((traj t).x_L - (traj t).x_i - d_des)
            * ((traj t).v_L - (traj t).v_i)) t :=
    hδ_sq.sub he_sq
  -- match the symbolic definitions that use ^2
  simpa [barrier, barrier_derivative, spacing_error, error_derivative,
         pow_two, sub_eq_add_neg, mul_comm] using h_barrier


lemma barrier_second_deriv_along_traj
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_δ' : Differentiable ℝ (deriv δ)) :
    ∀ t,
      HasDerivAt (fun τ => barrier_derivative δ (traj τ) d_des τ)
        (barrier_second_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) t := by
  intro t
  -- first derivatives of δ and δ'
  have hδ  : HasDerivAt δ (deriv δ t) t := (h_δ t).hasDerivAt
  have hδ' : HasDerivAt (deriv δ) (deriv (deriv δ) t) t := (h_δ' t).hasDerivAt
  -- kinematic first derivatives
  have hxL : HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t := (h_traj t).1
  have hxi : HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t := (h_traj t).2.2.1
  -- accelerations from the closed-loop hypothesis
  have hvL : HasDerivAt (fun τ => (traj τ).v_L) (a_L t) t := (h_traj t).2.1
  have hvi : HasDerivAt (fun τ => (traj τ).v_i)
      (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i t := (h_traj t).2.2.2.1
  -- ------------------------------------------------------------------
  -- derivative of 2·δ·δ'
  -- ------------------------------------------------------------------
  have h1 : HasDerivAt
      (fun τ => 2 * δ τ * deriv δ τ)
      (2 * (deriv δ t)^2 + 2 * δ t * deriv (deriv δ) t) t := by
    have h_prod : HasDerivAt (fun τ => δ τ * deriv δ τ)
        (deriv δ t * deriv δ t + δ t * deriv (deriv δ) t) t :=
      hδ.mul hδ'
    have h_prod2 : HasDerivAt (fun τ => δ τ * deriv δ τ)
        ((deriv δ t)^2 + δ t * deriv (deriv δ) t) t := by
      convert h_prod using 1
      · ring_nf
    convert h_prod2.const_mul 2 using 1
    · ext τ; ring_nf
    · rfl
    · ring_nf
    rw [mul_add]
    ring_nf
  -- ------------------------------------------------------------------
  -- e = x_L - x_i - d_des
  -- ------------------------------------------------------------------
  have he : HasDerivAt
      (fun τ => (traj τ).x_L - (traj τ).x_i - d_des)
      ((traj t).v_L - (traj t).v_i) t := by
    have h1 : HasDerivAt (fun τ => (traj τ).x_L - (traj τ).x_i)
        ((traj t).v_L - (traj t).v_i) t := hxL.sub hxi
    have h2 : HasDerivAt (fun _ => (d_des : ℝ)) 0 t := hasDerivAt_const t d_des
    have h3 : HasDerivAt
        (fun τ => (traj τ).x_L - (traj τ).x_i - d_des)
        ((traj t).v_L - (traj t).v_i - 0) t := h1.sub h2
    convert h3 using 1
    · ring
  -- ------------------------------------------------------------------
  -- ė = v_L - v_i
  -- ------------------------------------------------------------------
  have hde : HasDerivAt
      (fun τ => (traj τ).v_L - (traj τ).v_i)
      (a_L t - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i) t :=
    hvL.sub hvi
  -- ------------------------------------------------------------------
  -- derivative of e·ė
  -- ------------------------------------------------------------------
  have h_e_de : HasDerivAt
      (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                * ((traj τ).v_L - (traj τ).v_i))
      (((traj t).v_L - (traj t).v_i) * ((traj t).v_L - (traj t).v_i)
        + ((traj t).x_L - (traj t).x_i - d_des)
            * (a_L t - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i)) t :=
    he.mul hde
  -- ------------------------------------------------------------------
  -- derivative of -2·e·ė
  -- ------------------------------------------------------------------
  have h2 : HasDerivAt
      (fun τ => -2 * ((traj τ).x_L - (traj τ).x_i - d_des)
                    * ((traj τ).v_L - (traj τ).v_i))
      (-2 * (((traj t).v_L - (traj t).v_i)^2
            + ((traj t).x_L - (traj t).x_i - d_des)
                * (a_L t - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i))) t := by
    have h_prod : HasDerivAt
        (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                  * ((traj τ).v_L - (traj τ).v_i))
        (((traj t).v_L - (traj t).v_i)^2
          + ((traj t).x_L - (traj t).x_i - d_des)
              * (a_L t - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i)) t := by
      convert h_e_de using 1
      · ring_nf
    convert h_prod.const_mul (-2) using 1
    · ext τ; ring_nf
    · rfl
    ext τ
    ring_nf
  -- ------------------------------------------------------------------
  -- assemble 2·δ·δ' + (-2·e·ė)
  -- ------------------------------------------------------------------
  have h_sum := h1.add h2
  have h_fun : (fun τ => 2 * δ τ * deriv δ τ
                        - 2 * ((traj τ).x_L - (traj τ).x_i - d_des)
                            * ((traj τ).v_L - (traj τ).v_i))
              = fun τ => barrier_derivative δ (traj τ) d_des τ := by
    ext τ
    simp only [barrier_derivative, spacing_error, error_derivative]
    ring
  have h_sum' : HasDerivAt
      (fun τ => barrier_derivative δ (traj τ) d_des τ)
      (2 * (deriv δ t)^2 + 2 * δ t * deriv (deriv δ) t
        - 2 * ((traj t).v_L - (traj t).v_i)^2
        - 2 * ((traj t).x_L - (traj t).x_i - d_des)
            * (a_L t - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i)) t := by
    convert h_sum using 1
    · ext τ
      ring_nf
    · ext τ
      ring_nf
    · simp [barrier_derivative, spacing_error, error_derivative]
      rfl
    ring_nf
  have h_aL_eq :
    (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_L = a_L t := by
    unfold safe_control_input
    dsimp
    by_cases h1 : spacing_error (traj t) d_des < 0
    · simp [h1]
    · by_cases h2 : spacing_error (traj t) d_des > 0
      · simp [h1, h2]
      · simp [h1, h2]
  have h_sum'' : HasDerivAt
      (fun τ => barrier_derivative δ (traj τ) d_des τ)
      (2 * (deriv δ t)^2 + 2 * δ t * deriv (deriv δ) t
        - 2 * ((traj t).v_L - (traj t).v_i)^2
        - 2 * ((traj t).x_L - (traj t).x_i - d_des)
            * ((safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_L
                - (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i).a_i)) t := by
    convert h_sum' using 1
    · rw [h_aL_eq]
  -- match the definition of barrier_second_derivative
  simpa [barrier_second_derivative, error_second_derivative,
         spacing_error, error_derivative, sub_eq_add_neg, pow_two, mul_comm]
    using h_sum''


lemma psi0_deriv_along_traj
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) :
    ∀ t,
      HasDerivAt (fun τ => psi0 δ (traj τ) d_des τ)
        (barrier_derivative δ (traj t) d_des t) t := by
  intro t
  simpa [psi0] using
    barrier_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ t

def psi0_scaled
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (t : ℝ) : ℝ :=
  Real.exp (alpha1 * t) *
    psi0 δ (traj t) d_des t

def psi1_scaled
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (t : ℝ) : ℝ :=
  Real.exp (alpha2 * t) *
    psi1 δ (traj t) d_des t

lemma psi1_deriv_along_traj
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_δ' : Differentiable ℝ (deriv δ)) :
    ∀ t,
      HasDerivAt (fun τ => psi1 δ (traj τ) d_des τ)
        (psi1_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) t := by
  intro t
  -- ψ₁ = barrier_derivative + α₁ · barrier
  have h_bdot : HasDerivAt
      (fun τ => barrier_derivative δ (traj τ) d_des τ)
      (barrier_second_derivative δ (traj t) d_des
        (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) t :=
    barrier_second_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ h_δ' t
  have h_b : HasDerivAt
      (fun τ => barrier δ (traj τ) d_des τ)
      (barrier_derivative δ (traj t) d_des t) t :=
    barrier_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ t
  have h_αb : HasDerivAt
      (fun τ => alpha1 * barrier δ (traj τ) d_des τ)
      (alpha1 * barrier_derivative δ (traj t) d_des t) t :=
    h_b.const_mul alpha1
  -- sum
  have h_sum := h_bdot.add h_αb
  -- match the definition of psi1 / psi1_derivative
  unfold psi1 psi1_derivative
  exact h_sum

lemma psi1_scaled_differentiable
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_δ' : Differentiable ℝ (deriv δ)) :
    Differentiable ℝ (fun τ => psi1_scaled δ traj d_des τ) := by
  intro t
  -- exp is differentiable
  have h_exp : DifferentiableAt ℝ (fun τ => Real.exp (alpha2 * τ)) t := by
    fun_prop
  -- ψ₁ is differentiable (from the HasDerivAt we just proved)
  have h_psi1 : DifferentiableAt ℝ (fun τ => psi1 δ (traj τ) d_des τ) t :=
    (psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj h_δ h_δ' t).differentiableAt
  -- product
  exact h_exp.mul h_psi1

lemma psi0_scaled_differentiable
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) :
    Differentiable ℝ (fun τ => psi0_scaled δ traj d_des τ) := by
  intro t
  have h_exp : DifferentiableAt ℝ (fun τ => Real.exp (alpha1 * τ)) t := by
    fun_prop
  have h_psi0 : DifferentiableAt ℝ (fun τ => psi0 δ (traj τ) d_des τ) t :=
    (psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ t).differentiableAt
  exact h_exp.mul h_psi0


lemma closed_loop_psi1_differential_inequality
    (δ : ℝ → ℝ) (d_des : ℝ) (a_L : ℝ → ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) (h_δ' : Differentiable ℝ (deriv δ))
    (h_hocbf : ∀ t ≥ 0,
        HOCBF_Condition δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi1 δ (traj τ) d_des τ)
        (psi1_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) t
      ∧
      psi1_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
        + alpha2 * psi1 δ (traj t) d_des t ≥ 0 := by
  intro t ht
  constructor
  · exact psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj h_δ h_δ' t
  · exact h_hocbf t ht

lemma psi1_scaled_hasDerivAt
    (δ : ℝ → ℝ) (d_des : ℝ) (a_L : ℝ → ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) (h_δ' : Differentiable ℝ (deriv δ)) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi1_scaled δ traj d_des τ)
        (Real.exp (alpha2 * t) *
          (psi1_derivative δ (traj t) d_des
            (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
            + alpha2 * psi1 δ (traj t) d_des t)) t := by
  intro t ht
  have h_exp : HasDerivAt (fun τ => Real.exp (alpha2 * τ))
      (alpha2 * Real.exp (alpha2 * t)) t := by
    convert (Real.hasDerivAt_exp (alpha2 * t)).comp t
      (hasDerivAt_id t |>.const_mul alpha2) using 1
    · ext τ; ring_nf
    · ext τ; ring_nf
    · rfl
    ring_nf
  have h_psi : HasDerivAt (fun τ => psi1 δ (traj τ) d_des τ)
      (psi1_derivative δ (traj t) d_des
        (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t) t :=
    psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj h_δ h_δ' t
  have h_prod : HasDerivAt
    ((fun τ => Real.exp (alpha2 * τ)) * (fun τ => psi1 δ (traj τ) d_des τ))
    (alpha2 * Real.exp (alpha2 * t) * psi1 δ (traj t) d_des t
      + Real.exp (alpha2 * t) *
          (psi1_derivative δ (traj t) d_des
            (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t)) t :=
  h_exp.mul h_psi
  -- the two ways of writing the product are equal as functions
  have h_fun : (fun τ => Real.exp (alpha2 * τ) * psi1 δ (traj τ) d_des τ)
              = (fun τ => Real.exp (alpha2 * τ)) * (fun τ => psi1 δ (traj τ) d_des τ) := by
    rfl
  -- transport
  have h_scaled : HasDerivAt (fun τ => psi1_scaled δ traj d_des τ)
      (alpha2 * Real.exp (alpha2 * t) * psi1 δ (traj t) d_des t
        + Real.exp (alpha2 * t) *
            (psi1_derivative δ (traj t) d_des
              (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t)) t := by
    convert h_prod using 1
    · exact h_fun.symm
  -- finally rewrite the derivative expression into the factored form you want
  convert h_scaled using 1
  · ring_nf

lemma psi0_scaled_hasDerivAt
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi0_scaled δ traj d_des τ)
        (Real.exp (alpha1 * t) *
          (barrier_derivative δ (traj t) d_des t
            + alpha1 * psi0 δ (traj t) d_des t)) t := by
  intro t ht
  have h_exp : HasDerivAt (fun τ => Real.exp (alpha1 * τ))
      (alpha1 * Real.exp (alpha1 * t)) t := by
    convert (Real.hasDerivAt_exp (alpha1 * t)).comp t
      (hasDerivAt_id t |>.const_mul alpha1) using 1
    · ext τ; ring_nf
    · ext τ; ring_nf
    · rfl
    ring_nf
  have h_psi : HasDerivAt (fun τ => psi0 δ (traj τ) d_des τ)
      (barrier_derivative δ (traj t) d_des t) t :=
    psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ t
  have h_prod := h_exp.mul h_psi
  convert h_prod using 1
  · rfl
  · ext τ; ring_nf
  · rfl
  ring_nf

lemma psi1_scaled_deriv_nonneg
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_dyn : ∀ t ≥ 0,
        psi1_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
        + alpha2 * psi1 δ (traj t) d_des t ≥ 0) :
    ∀ t ≥ 0,
      0 ≤ Real.exp (alpha2 * t) *
            (psi1_derivative δ (traj t) d_des
              (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
              + alpha2 * psi1 δ (traj t) d_des t) := by
  intro t ht
  have h_exp_pos : 0 < Real.exp (alpha2 * t) := by positivity
  have h_ineq := h_dyn t ht
  exact mul_nonneg (le_of_lt h_exp_pos) h_ineq

lemma psi1_scaled_monotone
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ) (h_δ' : Differentiable ℝ (deriv δ))
    (h_dyn : ∀ t ≥ 0,
        psi1_derivative δ (traj t) d_des
          (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
        + alpha2 * psi1 δ (traj t) d_des t ≥ 0) :
    MonotoneOn (psi1_scaled δ traj d_des) (Set.Ici 0) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
  · intro x hx
    exact (psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj h_δ h_δ'
          x hx).differentiableAt.continuousAt.continuousWithinAt
  · intro x hx
    exact (psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj h_δ h_δ' x
    (le_of_lt (by simpa [interior_Ici] using hx))).differentiableAt.differentiableWithinAt
  · intro x hx
    have hx0 : 0 < x := by simpa [interior_Ici] using hx
    have h := psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj h_δ h_δ'
                x (le_of_lt hx0)
    have hnn := psi1_scaled_deriv_nonneg δ d_des traj nominal_a_i a_L h_dyn
                x (le_of_lt hx0)
    rw [h.deriv]
    exact hnn


lemma psi1_forward_invariant
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (h_mono :
      MonotoneOn
        (psi1_scaled δ traj d_des)
        (Set.Ici 0))
    (h_init :
      psi1 δ (traj 0) d_des 0 ≥ 0) :
    ∀ t ≥ 0,
      psi1 δ (traj t) d_des t ≥ 0 := by
  intro t ht
  have h_mono_t :
      psi1_scaled δ traj d_des 0 ≤
      psi1_scaled δ traj d_des t := by
    apply h_mono
    · simp
    · exact ht
    · exact ht
  have h_init_scaled :
      0 ≤ psi1_scaled δ traj d_des 0 := by
    rw [psi1_scaled]
    norm_num
    exact h_init
  have h_scaled_nonneg :
      0 ≤ psi1_scaled δ traj d_des t := by
    exact le_trans h_init_scaled h_mono_t
  have h_exp_pos :
      0 < Real.exp (alpha2 * t) := by
    positivity
  have h_exp_nonzero :
      Real.exp (alpha2 * t) ≠ 0 := ne_of_gt h_exp_pos
  have h_scaled_nonneg_exp :
      0 ≤ Real.exp (alpha2 * t) *
          psi1 δ (traj t) d_des t := by
    simpa [psi1_scaled] using h_scaled_nonneg
  have h_div :
      0 ≤
        (Real.exp (alpha2 * t) *
          psi1 δ (traj t) d_des t) /
          Real.exp (alpha2 * t) := by
    exact div_nonneg h_scaled_nonneg_exp (le_of_lt h_exp_pos)
  simpa [h_exp_nonzero] using h_div


lemma psi0_dynamics_nonneg
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_psi1 : ∀ t ≥ 0, psi1 δ (traj t) d_des t ≥ 0) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi0 δ (traj τ) d_des τ)
        (barrier_derivative δ (traj t) d_des t) t
      ∧
      barrier_derivative δ (traj t) d_des t
        + alpha1 * psi0 δ (traj t) d_des t ≥ 0 := by
  intro t ht
  constructor
  · exact psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj h_δ t
  · -- ψ₁ = ḣ + α₁·h ≥ 0
    simpa [psi1, psi0] using h_psi1 t ht

lemma psi0_scaled_deriv_nonneg
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (h_dyn : ∀ t ≥ 0,
        barrier_derivative δ (traj t) d_des t
        + alpha1 * psi0 δ (traj t) d_des t ≥ 0) :
    ∀ t ≥ 0,
      0 ≤ Real.exp (alpha1 * t) *
            (barrier_derivative δ (traj t) d_des t
              + alpha1 * psi0 δ (traj t) d_des t) := by
  intro t ht
  have h_exp_pos : 0 < Real.exp (alpha1 * t) := by positivity
  have h_ineq := h_dyn t ht
  exact mul_nonneg (le_of_lt h_exp_pos) h_ineq


lemma psi0_scaled_monotone
    (δ : ℝ → ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_dyn : ∀ t ≥ 0,
        barrier_derivative δ (traj t) d_des t
        + alpha1 * psi0 δ (traj t) d_des t ≥ 0) :
    MonotoneOn (psi0_scaled δ traj d_des) (Set.Ici 0) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
  · intro x hx
    exact (psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj h_δ x hx).differentiableAt.continuousAt.continuousWithinAt
  · intro x hx
    exact (psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj h_δ x (le_of_lt (by simpa [interior_Ici] using hx))).differentiableAt.differentiableWithinAt
  · intro x hx
    have hx0 : 0 ≤ x := le_of_lt (by simpa [interior_Ici] using hx)
    have h := psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj h_δ x hx0
    have hnn := psi0_scaled_deriv_nonneg δ d_des traj h_dyn x hx0
    rw [h.deriv]
    exact hnn


lemma psi0_forward_invariant
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (h_mono :
      MonotoneOn
        (psi0_scaled δ traj d_des)
        (Set.Ici 0))
    (h_init :
      psi0 δ (traj 0) d_des 0 ≥ 0) :
    ∀ t ≥ 0,
      psi0 δ (traj t) d_des t ≥ 0 := by
  intro t ht
  have h_mono_t :
      psi0_scaled δ traj d_des 0 ≤
      psi0_scaled δ traj d_des t := by
    apply h_mono
    · simp
    · exact ht
    · exact ht
  have h_init_scaled :
      0 ≤ psi0_scaled δ traj d_des 0 := by
    rw [psi0_scaled]
    norm_num
    exact h_init
  have h_scaled_nonneg :
      0 ≤ psi0_scaled δ traj d_des t := by
    exact le_trans h_init_scaled h_mono_t
  have h_exp_pos :
      0 < Real.exp (alpha1 * t) := by
    positivity
  have h_exp_nonzero :
      Real.exp (alpha1 * t) ≠ 0 := ne_of_gt h_exp_pos
  have h_scaled_nonneg_exp :
      0 ≤ Real.exp (alpha1 * t) *
          psi0 δ (traj t) d_des t := by
    simpa [psi0_scaled] using h_scaled_nonneg
  have h_div :
      0 ≤
        (Real.exp (alpha1 * t) *
          psi0 δ (traj t) d_des t) /
          Real.exp (alpha1 * t) := by
    exact div_nonneg h_scaled_nonneg_exp (le_of_lt h_exp_pos)
  simpa [h_exp_nonzero] using h_div


lemma barrier_forward_invariant
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (h_psi0 :
      ∀ t ≥ 0,
        psi0 δ (traj t) d_des t ≥ 0) :
    ∀ t ≥ 0,
      barrier δ (traj t) d_des t ≥ 0 := by
  intro t ht
  have h := h_psi0 t ht
  simpa [psi0] using h


theorem adaptive_hocbf_forward_invariance
    (δ : ℝ → ℝ)
    (d_des : ℝ)
    (traj : ℝ → VehicleState)
    (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_dyn : ∀ t ≥ 0,
        barrier_derivative δ (traj t) d_des t
          + alpha1 * psi0 δ (traj t) d_des t ≥ 0)
    (h_init : initial_hocbf_conditions δ (traj 0) d_des) :
    ∀ t ≥ 0, barrier δ (traj t) d_des t ≥ 0 := by
  have h_psi0_mono :
      MonotoneOn (psi0_scaled δ traj d_des) (Set.Ici 0) :=
    psi0_scaled_monotone δ d_des traj nominal_a_i a_L h_traj h_δ h_dyn
  have h_psi0_forward :
      ∀ t ≥ 0, psi0 δ (traj t) d_des t ≥ 0 :=
    fun t ht => psi0_forward_invariant δ traj d_des h_psi0_mono h_init.1 t ht
  exact barrier_forward_invariant δ traj d_des h_psi0_forward


theorem adaptive_closed_loop_spacing_safety
    (δ : ℝ → ℝ)
    (traj : ℝ → VehicleState)
    (d_des : ℝ)
    (h_barrier :
      ∀ t ≥ 0,
        barrier δ (traj t) d_des t ≥ 0)
    (h_delta_nonneg :
      ∀ t ≥ 0,
        0 ≤ δ t) :
    ∀ t ≥ 0,
      |spacing_error (traj t) d_des| ≤ δ t := by
    intro t ht
    have h_barrier_t :=
      h_barrier t ht
    have h_delta_t :=
      h_delta_nonneg t ht
    have h_abs_delta :
        |δ t| = δ t := by
      exact abs_of_nonneg h_delta_t
    have h_sq :
        (spacing_error (traj t) d_des)^2 ≤ (δ t)^2 := by
      unfold barrier at h_barrier_t
      nlinarith
    have h_abs_bound :
        |spacing_error (traj t) d_des| ≤ |δ t| := by
      exact sq_le_sq.mp h_sq
    simpa [h_abs_delta] using h_abs_bound


/-! ## Step 4 – End-to-end adaptive spacing safety -/

theorem adaptive_spacing_safety
    (δ : ℝ → ℝ)
    (params : DeltaParams)
    (h_adapt : delta_implements_adaptive_law params δ)
    (d_des : ℝ)
    (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_δ : Differentiable ℝ δ)
    (h_δ' : Differentiable ℝ (deriv δ))
    (h_vel : ∀ t ≥ 0, ZeroErrorVelocityBound params (traj t))
    (h_gain : ZeroErrorGainCondition params)
    (h_init : initial_hocbf_conditions δ (traj 0) d_des)
    (h_delta_nonneg : ∀ t ≥ 0, 0 ≤ δ t) :
    ∀ t ≥ 0, |spacing_error (traj t) d_des| ≤ δ t := by
  -- 1. The closed-loop controller satisfies the HOCBF inequality at every instant
  have h_hocbf : ∀ t ≥ 0,
      HOCBF_Condition δ (traj t) d_des
        (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t :=
    closed_loop_hocbf_along_trajectory δ params h_adapt d_des traj a_L nominal_a_i h_vel h_gain
  -- 2. Consequently the highest-order differential inequality holds
  have h_psi1_dyn : ∀ t ≥ 0,
      psi1_derivative δ (traj t) d_des
        (safe_control_input δ (traj t) d_des t (a_L t) nominal_a_i) t
        + alpha2 * psi1 δ (traj t) d_des t ≥ 0 := by
    intro t ht
    exact (h_hocbf t ht)
  -- 3. ψ₁ stays non-negative (exponential weighting + monotonicity)
  have h_psi1_nonneg : ∀ t ≥ 0, psi1 δ (traj t) d_des t ≥ 0 := by
    have h_mono :=
      psi1_scaled_monotone δ d_des traj nominal_a_i a_L h_traj h_δ h_δ' h_psi1_dyn
    exact fun t ht =>
      psi1_forward_invariant δ traj d_des h_mono h_init.2 t ht
  -- 4. The first-order inequality for ψ₀ follows
  have h_psi0_dyn : ∀ t ≥ 0,
      barrier_derivative δ (traj t) d_des t
        + alpha1 * psi0 δ (traj t) d_des t ≥ 0 := by
    intro t ht
    have h := psi0_dynamics_nonneg δ d_des traj nominal_a_i a_L h_traj h_δ h_psi1_nonneg t ht
    exact h.2
  -- 5. ψ₀ (i.e. the barrier) stays non-negative
  have h_barrier_nonneg : ∀ t ≥ 0, barrier δ (traj t) d_des t ≥ 0 := by
    have h_mono :=
      psi0_scaled_monotone δ d_des traj nominal_a_i a_L h_traj h_δ h_psi0_dyn
    have h_psi0 :=
      fun t ht => psi0_forward_invariant δ traj d_des h_mono h_init.1 t ht
    exact barrier_forward_invariant δ traj d_des h_psi0
  -- 6. Non-negativity of the barrier + non-negativity of δ ⇒ spacing safety
  exact adaptive_closed_loop_spacing_safety δ traj d_des h_barrier_nonneg h_delta_nonneg



end
end DBCCProject
