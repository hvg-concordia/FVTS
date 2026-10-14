import Mathlib

namespace DBCCProject
set_option linter.style.longLine false
noncomputable section

-- # Fixed-tolerance (constant-δ) HOCBF formalization
--
-- This file is the constant-δ specialization of the time-varying development
-- `DBCCProject`.  The spacing barrier is
--
-- h = δ² − e²,     e = (x_L − x_i) − d_des,
--
-- with a **fixed** design tolerance δ > 0.  Consequently
--
-- δ̇ ≡ 0,   δ̈ ≡ 0,
--
-- the explicit time argument disappears from h, and the adaptive-law
-- hypotheses (`M_rate`, `M2`, `LocallyLipschitz δ`, `delta_implements_adaptive_law`)
-- are dropped.
--
-- Relative-degree-2 HOCBF auxiliaries (linear class-𝒦 gains α₁, α₂):
--
-- ψ₀ = h,
-- ψ₁ = ḣ + α₁ h,
-- HOCBF condition:  ψ̇₁ + α₂ ψ₁ ≥ 0.
--
--



-- ## 1. Core Structures


structure VehicleState where
  x_L : ℝ
  v_L : ℝ
  x_i : ℝ
  v_i : ℝ

structure VehicleInput where
  a_L : ℝ
  a_i : ℝ

-- Parameters that remain after δ is frozen.
-- Only a positive constant tolerance and a bound on |ė| at e = 0
-- are required (the latter solely for the measure-zero e = 0 slice).

structure FixedDeltaParams where
  δ : ℝ
  M_vel : ℝ
  h_delta_pos : 0 < δ
  h_Mvel_pos  : 0 < M_vel


-- ## 2. Physical Bounds


def a_min : ℝ := -8
def a_max : ℝ := 3

def ValidInput (input : VehicleInput) : Prop :=
  a_min ≤ input.a_L ∧ input.a_L ≤ a_max ∧
  a_min ≤ input.a_i ∧ input.a_i ≤ a_max

def ValidState (state : VehicleState) : Prop :=
  state.x_L > state.x_i ∧
  0 < state.v_L ∧ state.v_L < 100 ∧
  0 < state.v_i ∧ state.v_i < 100

lemma a_min_le_zero : a_min ≤ 0 := by
  norm_num [a_min]

lemma zero_le_a_max : (0 : ℝ) ≤ a_max := by
  norm_num [a_max]


-- ## 3. Spacing Error (Continuous Time)


-- e = (x_L − x_i) − d_des.  No vehicle-length offset.

def spacing_error (state : VehicleState) (d_des : ℝ) : ℝ :=
  (state.x_L - state.x_i) - d_des

-- ė = v_L − v_i.

def error_derivative (state : VehicleState) : ℝ :=
  state.v_L - state.v_i

-- ë = a_L − a_i  (control appears here: relative degree 2).

def error_second_derivative (input : VehicleInput) : ℝ :=
  input.a_L - input.a_i


-- ## 4. Class-𝒦 Gains and Constant Tolerance


def alpha1 : ℝ := 1.0
def alpha2 : ℝ := 1.0

lemma alpha1_pos : 0 < alpha1 := by norm_num [alpha1]
lemma alpha2_pos : 0 < alpha2 := by norm_num [alpha2]
lemma alpha1_nonneg : 0 ≤ alpha1 := le_of_lt alpha1_pos
lemma alpha2_nonneg : 0 ≤ alpha2 := le_of_lt alpha2_pos
lemma alpha1_alpha2_nonneg : 0 ≤ alpha1 * alpha2 := by
  exact mul_nonneg alpha1_nonneg alpha2_nonneg


-- ## 5. Barrier Function (Constant δ)


-- h(x) = δ² − e².  Independent of t once δ is fixed.
--
def barrier (δ : ℝ) (state : VehicleState) (d_des : ℝ) : ℝ :=
  δ ^ 2 - (spacing_error state d_des) ^ 2

def SafeSet (δ : ℝ) (state : VehicleState) (d_des : ℝ) : Prop :=
  barrier δ state d_des ≥ 0


-- ## 6. Barrier Derivatives (δ̇ = δ̈ = 0)


-- ḣ = ∂h/∂e · ė = (−2e) · ė.
--
-- The explicit-time term 2 δ δ̇ of the time-varying theory is identically zero.
--

def barrier_derivative (state : VehicleState) (d_des : ℝ) : ℝ :=
  let e := spacing_error state d_des
  let de := error_derivative state
  (-2 * e * de)



def barrier_second_derivative (state : VehicleState) (d_des : ℝ) (input : VehicleInput) : ℝ :=
  let e  := spacing_error state d_des
  let de := error_derivative state
  let dde := error_second_derivative input
  (-2 * (de ^ 2) - (2 * e * dde))


-- ## 7. Higher-Order CBF Auxiliaries
--


def psi0 (δ : ℝ) (state : VehicleState) (d_des : ℝ) : ℝ :=
  barrier δ state d_des

-- ψ₁ = ḣ + α₁ h.

def psi1 (δ : ℝ) (state : VehicleState) (d_des : ℝ) : ℝ :=
  barrier_derivative state d_des + alpha1 * psi0 δ state d_des

-- ψ̇₁ = ḧ + α₁ ḣ.

def psi1_derivative (state : VehicleState) (d_des : ℝ) (input : VehicleInput) : ℝ :=
  barrier_second_derivative state d_des input
    + alpha1 * barrier_derivative state d_des

-- HOCBF inequality: ψ̇₁ + α₂ ψ₁ ≥ 0.

def HOCBF_Condition
    (δ : ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) : Prop :=
  psi1_derivative state d_des input + alpha2 * psi1 δ state d_des ≥ 0


-- ## 8. Exact Control Bounds
--
-- Algebra (shown in `FixedDelta_HOCBF_Derivation.md`):
--
-- ψ̇₁ + α₂ ψ₁
-- = 2 e (a_i − a_L) + 2 N,
--
-- N := −ė² − (α₁+α₂) e ė + (α₁ α₂ / 2) h.
--
-- Hence
-- e > 0  ⇒  a_i ≥ a_L − N/e,
-- e < 0  ⇒  a_i ≤ a_L − N/e,
-- e = 0  ⇒  N ≥ 0  (independent of a_i).
--


def hocbf_num (δ : ℝ) (state : VehicleState) (d_des : ℝ) : ℝ :=
  let e  := spacing_error state d_des
  let de := error_derivative state
  let h  := barrier δ state d_des
  (-(de ^ 2)) - (alpha1 + alpha2) * e * de + (alpha1 * alpha2 / 2) * h


def control_lower_bound (δ : ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) : ℝ :=
  input.a_L - (hocbf_num δ state d_des) / (spacing_error state d_des)



def control_upper_bound
    (δ : ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput) : ℝ :=
  input.a_L - (hocbf_num δ state d_des) / (spacing_error state d_des)


-- ## 9. Saturated Safe Controller


lemma min_upper_nominal_valid
    (upper nominal_a_i : ℝ)
    (h_upper_lower : a_min ≤ upper)
    (h_nominal_lower : a_min ≤ nominal_a_i)
    (h_upper_max : upper ≤ a_max) :
    a_min ≤ min upper nominal_a_i ∧ min upper nominal_a_i ≤ a_max := by
  constructor
  · exact le_min h_upper_lower h_nominal_lower
  · exact le_trans (min_le_left upper nominal_a_i) h_upper_max

lemma max_lower_nominal_valid
    (lower nominal_a_i : ℝ)
    (h_lower_min : a_min ≤ lower)
    (h_lower_max : lower ≤ a_max)
    (h_nominal_max : nominal_a_i ≤ a_max) :
    a_min ≤ max lower nominal_a_i ∧ max lower nominal_a_i ≤ a_max := by
  constructor
  · exact le_trans h_lower_min (le_max_left lower nominal_a_i)
  · exact max_le h_lower_max h_nominal_max

-- QP-style filter: enforce the HOCBF half-space and keep the nominal
-- command whenever it is already safe.  At e = 0 the constraint does
-- not involve a_i, so the nominal command is passed through.

def safe_control_input
    (δ : ℝ) (state : VehicleState) (d_des : ℝ)
    (a_L : ℝ) (nominal_a_i : ℝ) : VehicleInput :=
  let lower := control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i }
  let upper := control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i }
  let e := spacing_error state d_des
  if e < 0 then
    { a_L := a_L, a_i := min upper nominal_a_i }
  else if e > 0 then
    { a_L := a_L, a_i := max lower nominal_a_i }
  else
    { a_L := a_L, a_i := nominal_a_i }

lemma safe_control_input_valid
    (δ : ℝ) (state : VehicleState) (d_des : ℝ)
    (a_L : ℝ) (nominal_a_i : ℝ)
    (h_aL : a_min ≤ a_L ∧ a_L ≤ a_max)
    (h_nominal : a_min ≤ nominal_a_i ∧ nominal_a_i ≤ a_max)
    (h_upper_lower :
      a_min ≤ control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i })
    (h_upper_max :
      control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } ≤ a_max)
    (h_lower_min :
      a_min ≤ control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i })
    (h_lower_max :
      control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i } ≤ a_max) :
    ValidInput (safe_control_input δ state d_des a_L nominal_a_i) := by
  unfold ValidInput safe_control_input
  dsimp
  by_cases h_neg : spacing_error state d_des < 0
  · simp only [h_neg, if_pos]
    have h :=
      min_upper_nominal_valid
        (control_upper_bound δ state d_des { a_L := a_L, a_i := nominal_a_i })
        nominal_a_i h_upper_lower h_nominal.1 h_upper_max
    constructor
    · exact h_aL.1
    constructor
    · exact h_aL.2
    exact h
  · by_cases h_pos : spacing_error state d_des > 0
    · simp only [h_neg, h_pos]
      have h :=
        max_lower_nominal_valid
          (control_lower_bound δ state d_des { a_L := a_L, a_i := nominal_a_i })
          nominal_a_i h_lower_min h_lower_max h_nominal.2
      constructor
      · exact h_aL.1
      constructor
      · exact h_aL.2
      exact h
    · simp only [h_neg, h_pos]
      constructor
      · exact h_aL.1
      constructor
      · exact h_aL.2
      constructor
      · exact h_nominal.1
      · exact h_nominal.2


-- ## 10. Equivalence: control bound ⇔ HOCBF inequality


theorem control_lower_bound_implies_HOCBF
    (δ : ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput)
    (h_e : spacing_error state d_des > 0)
    (h_bound : input.a_i ≥ control_lower_bound δ state d_des input) :
    HOCBF_Condition δ state d_des input := by
  unfold HOCBF_Condition
  have h_eq :
      psi1_derivative state d_des input + alpha2 * psi1 δ state d_des
        =
      2 * spacing_error state d_des * (input.a_i - input.a_L)
        + 2 * hocbf_num δ state d_des := by
    unfold psi1_derivative psi1 psi0
      barrier_second_derivative barrier_derivative hocbf_num barrier
      error_second_derivative error_derivative spacing_error
    field_simp [ne_of_gt h_e]
    ring_nf
  rw [h_eq]
  set e := spacing_error state d_des
  set num := hocbf_num δ state d_des
  have he_pos : 0 < e := h_e
  have h1 : input.a_i - input.a_L ≥ - num / e := by
    have hb : input.a_i ≥ input.a_L - num / e := by
      simpa [control_lower_bound, num] using h_bound
    rw [div_eq_mul_inv]
    have hsub := sub_le_sub_right (ge_iff_le.mp hb) input.a_L
    -- hsub : (input.a_L - num * e⁻¹) - input.a_L ≤ input.a_i - input.a_L
    rw [div_eq_mul_inv] at hsub
    ring_nf at hsub
    convert hsub using 1 <;> ring
  have h2 : e * (input.a_i - input.a_L) ≥ - num := by
    have hmul :=
      mul_le_mul_of_nonneg_left (ge_iff_le.mp h1) (le_of_lt he_pos)
    have he_ne : e ≠ 0 := ne_of_gt he_pos
    field_simp [he_ne] at hmul
    exact hmul
  linarith



theorem control_upper_bound_implies_HOCBF
    (δ : ℝ) (state : VehicleState) (d_des : ℝ) (input : VehicleInput)
    (h_e : spacing_error state d_des < 0)
    (h_bound : input.a_i ≤ control_upper_bound δ state d_des input) :
    HOCBF_Condition δ state d_des input := by
  unfold HOCBF_Condition psi1 psi1_derivative
  unfold barrier_derivative barrier_second_derivative psi0 barrier
  simp only [spacing_error, error_derivative, error_second_derivative]
  set e   := spacing_error state d_des with he
  set de  := error_derivative state with hde
  set num := hocbf_num δ state d_des with hnum
  have h1 : input.a_i - input.a_L ≤ - num / e := by
    have : input.a_i ≤ input.a_L - num / e := by
      simpa [control_upper_bound, he, hnum] using h_bound
    ring_nf at this
    simp [he, hnum]
    ring_nf
    linarith [this]
  have he_neg : e < 0 := by simpa [he] using h_e
  have h2 : e * (input.a_i - input.a_L) ≥ - num := by
    have := mul_le_mul_of_nonpos_left h1 (le_of_lt he_neg)
    field_simp [ne_of_lt he_neg] at this
    simpa [ne_of_lt he_neg] using this
  have h_target : 0 ≤ 2 * e * (input.a_i - input.a_L) + 2 * num := by
    have h3 : 2 * e * (input.a_i - input.a_L) ≥ -2 * num := by
      linarith [h2]
    linarith [h3]
  convert h_target using 1
  · simp only [num, he]
    unfold spacing_error hocbf_num
    simp only [spacing_error, error_derivative, barrier]
    ring_nf


-- ## 11. The e = 0 case
--
-- At e = 0 the input a_i drops out of the HOCBF inequality.  Substituting
-- the constant-δ formulae yields the purely kinematic restriction
--
-- α₁ α₂ δ² ≥ 2 ė².
--
-- A uniform bound |ė| ≤ M_vel therefore gives a numerical gain condition
-- that no longer involves M_rate or M₂.
--


def initial_hocbf_conditions
    (δ : ℝ) (state0 : VehicleState) (d_des : ℝ) : Prop :=
  psi0 δ state0 d_des ≥ 0 ∧ psi1 δ state0 d_des ≥ 0

def ZeroErrorVelocityBound (params : FixedDeltaParams) (state : VehicleState) : Prop :=
  |state.v_L - state.v_i| ≤ params.M_vel

-- Gain condition at e = 0:  α₁ α₂ δ² ≥ 2 M_vel².
-- Recovers the time-varying condition after setting M_rate = M₂ = 0
-- and δ_base = δ_max = δ.

def ZeroErrorGainCondition (params : FixedDeltaParams) : Prop :=
  alpha1 * alpha2 * params.δ ^ 2 ≥ 2 * params.M_vel ^ 2

theorem zero_error_implies_HOCBF
    (params : FixedDeltaParams)
    (state : VehicleState)
    (d_des : ℝ)
    (input : VehicleInput)
    (h_e_zero : spacing_error state d_des = 0)
    (h_vel : ZeroErrorVelocityBound params state)
    (h_gain : ZeroErrorGainCondition params) :
    HOCBF_Condition params.δ state d_des input := by
  unfold HOCBF_Condition psi1 psi1_derivative
    barrier_derivative barrier_second_derivative psi0 barrier
  simp only [error_derivative, error_second_derivative, spacing_error]
  -- After e = 0:  ψ̇₁ + α₂ ψ₁ = −2 ė² + α₁ α₂ δ²
  have h_vel_sq : (state.v_L - state.v_i) ^ 2 ≤ params.M_vel ^ 2 := by
    have hvel :
        -params.M_vel ≤ state.v_L - state.v_i ∧
        state.v_L - state.v_i ≤ params.M_vel :=
      abs_le.mp h_vel
    have hM : 0 ≤ params.M_vel := le_of_lt params.h_Mvel_pos
    nlinarith
  have h_main :
      0 ≤ -2 * (state.v_L - state.v_i) ^ 2 + alpha1 * alpha2 * params.δ ^ 2 := by
    unfold ZeroErrorGainCondition at h_gain
    nlinarith [h_vel_sq, h_gain]
  -- expand the HOCBF expression and match
  have he : state.x_L - state.x_i - d_des = 0 := by
    simpa [spacing_error] using h_e_zero
  simp [he]
  linarith [h_main]


-- ## 12. Closed-Loop HOCBF Satisfaction


theorem closed_loop_satisfies_hocbf
    (params : FixedDeltaParams)
    (d_des : ℝ) (state : VehicleState)
    (a_L : ℝ) (nominal_a_i : ℝ)
    (h_vel : ZeroErrorVelocityBound params state)
    (h_gain : ZeroErrorGainCondition params) :
    HOCBF_Condition params.δ state d_des
      (safe_control_input params.δ state d_des a_L nominal_a_i) := by
  by_cases h_neg : spacing_error state d_des < 0
  · have h_safe :
        safe_control_input params.δ state d_des a_L nominal_a_i =
          { a_L := a_L,
            a_i := min (control_upper_bound params.δ state d_des
                          { a_L := a_L, a_i := nominal_a_i })
                      nominal_a_i } := by
      unfold safe_control_input
      simp only [h_neg, if_pos]
    rw [h_safe]
    apply control_upper_bound_implies_HOCBF
    · exact h_neg
    · exact min_le_left _ _
  · by_cases h_pos : spacing_error state d_des > 0
    · have h_safe :
          safe_control_input params.δ state d_des a_L nominal_a_i =
            { a_L := a_L,
              a_i := max (control_lower_bound params.δ state d_des
                            { a_L := a_L, a_i := nominal_a_i })
                        nominal_a_i } := by
        unfold safe_control_input
        simp [h_neg, h_pos]
      rw [h_safe]
      apply control_lower_bound_implies_HOCBF
      · exact h_pos
      · exact le_max_left _ _
    · have h_zero : spacing_error state d_des = 0 := by
        apply le_antisymm <;> linarith
      have h_safe :
          safe_control_input params.δ state d_des a_L nominal_a_i =
            { a_L := a_L, a_i := nominal_a_i } := by
        unfold safe_control_input
        simp [h_zero, lt_self_iff_false]
      rw [h_safe]
      exact zero_error_implies_HOCBF params state d_des
        { a_L := a_L, a_i := nominal_a_i } h_zero h_vel h_gain


-- ## 13. Closed-Loop Trajectories


def is_closed_loop_trajectory
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ) : Prop :=
  ∀ t,
    HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t ∧
    HasDerivAt (fun τ => (traj τ).v_L) (a_L t) t ∧
    HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t ∧
    HasDerivAt (fun τ => (traj τ).v_i)
      (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i).a_i t ∧
    (a_min ≤ a_L t ∧ a_L t ≤ a_max)

lemma closed_loop_hocbf_along_trajectory
    (params : FixedDeltaParams)
    (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_vel : ∀ t ≥ 0, ZeroErrorVelocityBound params (traj t))
    (h_gain : ZeroErrorGainCondition params) :
    ∀ t ≥ 0,
      HOCBF_Condition params.δ (traj t) d_des
        (safe_control_input params.δ (traj t) d_des (a_L t) nominal_a_i) := by
  intro t ht
  exact closed_loop_satisfies_hocbf params d_des (traj t) (a_L t) nominal_a_i
    (h_vel t ht) h_gain


-- ## 14. Chain-rule lemmas along closed-loop trajectories
--
-- No differentiability hypothesis on δ is required: a constant is C^∞.
--


lemma barrier_deriv_along_traj
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t, HasDerivAt (fun τ => barrier δ (traj τ) d_des)
          (barrier_derivative (traj t) d_des) t := by
  intro t
  have hxL : HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t := (h_traj t).1
  have hxi : HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t := (h_traj t).2.2.1
  -- e = x_L − x_i − d_des
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
  -- e · e
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
  -- constant δ² has derivative 0
  have hδ_sq : HasDerivAt (fun _ => δ * δ) 0 t := hasDerivAt_const t (δ * δ)
  have h_barrier : HasDerivAt
      (fun τ => δ * δ
                - ((traj τ).x_L - (traj τ).x_i - d_des)
                    * ((traj τ).x_L - (traj τ).x_i - d_des))
      (0 - 2 * ((traj t).x_L - (traj t).x_i - d_des)
            * ((traj t).v_L - (traj t).v_i)) t :=
    hδ_sq.sub he_sq
  simpa [barrier, barrier_derivative, spacing_error, error_derivative,
         pow_two, sub_eq_add_neg, mul_comm] using h_barrier

lemma barrier_second_deriv_along_traj
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t,
      HasDerivAt (fun τ => barrier_derivative (traj τ) d_des)
        (barrier_second_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) t := by
  intro t
  have hxL : HasDerivAt (fun τ => (traj τ).x_L) (traj t).v_L t := (h_traj t).1
  have hxi : HasDerivAt (fun τ => (traj τ).x_i) (traj t).v_i t := (h_traj t).2.2.1
  have hvL : HasDerivAt (fun τ => (traj τ).v_L) (a_L t) t := (h_traj t).2.1
  have hvi : HasDerivAt (fun τ => (traj τ).v_i)
      (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i).a_i t :=
    (h_traj t).2.2.2.1
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
  have hde : HasDerivAt
      (fun τ => (traj τ).v_L - (traj τ).v_i)
      (a_L t - (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i).a_i) t :=
    hvL.sub hvi
  -- derivative of −2 e ė
  have h_e_de : HasDerivAt
      (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                * ((traj τ).v_L - (traj τ).v_i))
      (((traj t).v_L - (traj t).v_i) * ((traj t).v_L - (traj t).v_i)
        + ((traj t).x_L - (traj t).x_i - d_des)
            * (a_L t - (safe_control_input δ (traj t) d_des
                          (a_L t) nominal_a_i).a_i)) t :=
    he.mul hde
  have h2 : HasDerivAt
      (fun τ => -2 * ((traj τ).x_L - (traj τ).x_i - d_des)
                    * ((traj τ).v_L - (traj τ).v_i))
      (-2 * (((traj t).v_L - (traj t).v_i) ^ 2
            + ((traj t).x_L - (traj t).x_i - d_des)
                * (a_L t - (safe_control_input δ (traj t) d_des
                              (a_L t) nominal_a_i).a_i))) t := by
    have h_prod : HasDerivAt
        (fun τ => ((traj τ).x_L - (traj τ).x_i - d_des)
                  * ((traj τ).v_L - (traj τ).v_i))
        (((traj t).v_L - (traj t).v_i) ^ 2
          + ((traj t).x_L - (traj t).x_i - d_des)
              * (a_L t - (safe_control_input δ (traj t) d_des
                            (a_L t) nominal_a_i).a_i)) t := by
      convert h_e_de using 1
      · ring_nf
    convert h_prod.const_mul (-2) using 1
    · ext τ; ring_nf
    · rfl
    ext τ
    ring_nf
  have h_aL_eq :
      (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i).a_L = a_L t := by
    unfold safe_control_input
    dsimp
    by_cases h1 : spacing_error (traj t) d_des < 0
    · simp [h1]
    · by_cases h2 : spacing_error (traj t) d_des > 0
      · simp [h1, h2]
      · simp [h1, h2]
  have h_sum'' : HasDerivAt
      (fun τ => -2 * ((traj τ).x_L - (traj τ).x_i - d_des)
                    * ((traj τ).v_L - (traj τ).v_i))
      (-2 * ((traj t).v_L - (traj t).v_i) ^ 2
        - 2 * ((traj t).x_L - (traj t).x_i - d_des)
            * ((safe_control_input δ (traj t) d_des (a_L t) nominal_a_i).a_L
                - (safe_control_input δ (traj t) d_des
                    (a_L t) nominal_a_i).a_i)) t := by
    convert h2 using 1
    rw [h_aL_eq]
    field_simp [safe_control_input, spacing_error, control_lower_bound]
    ring_nf
  simpa [barrier_derivative, barrier_second_derivative, error_second_derivative,
         spacing_error, error_derivative, sub_eq_add_neg, pow_two, mul_comm]
    using h_sum''


lemma psi0_deriv_along_traj
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t,
      HasDerivAt (fun τ => psi0 δ (traj τ) d_des)
        (barrier_derivative (traj t) d_des) t := by
  intro t
  simpa [psi0] using
    barrier_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t

def psi0_scaled
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  Real.exp (alpha1 * t) * psi0 δ (traj t) d_des

def psi1_scaled
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ) (t : ℝ) : ℝ :=
  Real.exp (alpha2 * t) * psi1 δ (traj t) d_des

lemma psi1_deriv_along_traj
    (δ : ℝ) (d_des : ℝ) (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t,
      HasDerivAt (fun τ => psi1 δ (traj τ) d_des)
        (psi1_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) t := by
  intro t
  have h_bdot : HasDerivAt
      (fun τ => barrier_derivative (traj τ) d_des)
      (barrier_second_derivative (traj t) d_des
        (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) t :=
    barrier_second_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t
  have h_b : HasDerivAt
      (fun τ => barrier δ (traj τ) d_des)
      (barrier_derivative (traj t) d_des) t :=
    barrier_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t
  have h_αb : HasDerivAt
      (fun τ => alpha1 * barrier δ (traj τ) d_des)
      (alpha1 * barrier_derivative (traj t) d_des) t :=
    h_b.const_mul alpha1
  have h_sum := h_bdot.add h_αb
  unfold psi1 psi1_derivative
  exact h_sum

lemma psi1_scaled_differentiable
    (δ : ℝ) (d_des : ℝ) (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    Differentiable ℝ (fun τ => psi1_scaled δ traj d_des τ) := by
  intro t
  have h_exp : DifferentiableAt ℝ (fun τ => Real.exp (alpha2 * τ)) t := by
    fun_prop
  have h_psi1 : DifferentiableAt ℝ (fun τ => psi1 δ (traj τ) d_des) t :=
    (psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj t).differentiableAt
  exact h_exp.mul h_psi1

lemma psi0_scaled_differentiable
    (δ : ℝ) (d_des : ℝ) (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    Differentiable ℝ (fun τ => psi0_scaled δ traj d_des τ) := by
  intro t
  have h_exp : DifferentiableAt ℝ (fun τ => Real.exp (alpha1 * τ)) t := by
    fun_prop
  have h_psi0 : DifferentiableAt ℝ (fun τ => psi0 δ (traj τ) d_des) t :=
    (psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t).differentiableAt
  exact h_exp.mul h_psi0

lemma closed_loop_psi1_differential_inequality
    (δ : ℝ) (d_des : ℝ) (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_hocbf : ∀ t ≥ 0,
        HOCBF_Condition δ (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi1 δ (traj τ) d_des)
        (psi1_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) t
      ∧
      psi1_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)
        + alpha2 * psi1 δ (traj t) d_des ≥ 0 := by
  intro t ht
  constructor
  · exact psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj t
  · exact h_hocbf t ht

lemma psi1_scaled_hasDerivAt
    (δ : ℝ) (d_des : ℝ) (a_L : ℝ → ℝ)
    (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi1_scaled δ traj d_des τ)
        (Real.exp (alpha2 * t) *
          (psi1_derivative (traj t) d_des
            (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)
            + alpha2 * psi1 δ (traj t) d_des)) t := by
  intro t ht
  have h_exp : HasDerivAt (fun τ => Real.exp (alpha2 * τ))
      (alpha2 * Real.exp (alpha2 * t)) t := by
    convert (Real.hasDerivAt_exp (alpha2 * t)).comp t
      (hasDerivAt_id t |>.const_mul alpha2) using 1
    · ext τ; ring_nf
    · ext τ; ring_nf
    · rfl
    ring_nf
  have h_psi : HasDerivAt (fun τ => psi1 δ (traj τ) d_des)
      (psi1_derivative (traj t) d_des
        (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)) t :=
    psi1_deriv_along_traj δ d_des a_L traj nominal_a_i h_traj t
  have h_prod : HasDerivAt
    ((fun τ => Real.exp (alpha2 * τ)) * (fun τ => psi1 δ (traj τ) d_des))
    (alpha2 * Real.exp (alpha2 * t) * psi1 δ (traj t) d_des
      + Real.exp (alpha2 * t) *
          (psi1_derivative (traj t) d_des
            (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i))) t :=
    h_exp.mul h_psi
  have h_fun : (fun τ => Real.exp (alpha2 * τ) * psi1 δ (traj τ) d_des)
              = (fun τ => Real.exp (alpha2 * τ)) * (fun τ => psi1 δ (traj τ) d_des) := by
    rfl
  have h_scaled : HasDerivAt (fun τ => psi1_scaled δ traj d_des τ)
      (alpha2 * Real.exp (alpha2 * t) * psi1 δ (traj t) d_des
        + Real.exp (alpha2 * t) *
            (psi1_derivative (traj t) d_des
              (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i))) t := by
    convert h_prod using 1
    · exact h_fun.symm
  convert h_scaled using 1
  · ring_nf

lemma psi0_scaled_hasDerivAt
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ) (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi0_scaled δ traj d_des τ)
        (Real.exp (alpha1 * t) *
          (barrier_derivative (traj t) d_des
            + alpha1 * psi0 δ (traj t) d_des)) t := by
  intro t ht
  have h_exp : HasDerivAt (fun τ => Real.exp (alpha1 * τ))
      (alpha1 * Real.exp (alpha1 * t)) t := by
    convert (Real.hasDerivAt_exp (alpha1 * t)).comp t
      (hasDerivAt_id t |>.const_mul alpha1) using 1
    · ext τ; ring_nf
    · ext τ; ring_nf
    · rfl
    ring_nf
  have h_psi : HasDerivAt (fun τ => psi0 δ (traj τ) d_des)
      (barrier_derivative (traj t) d_des) t :=
    psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t
  have h_prod := h_exp.mul h_psi
  convert h_prod using 1
  · rfl
  · ext τ; ring_nf
  · rfl
  ring_nf

lemma psi1_scaled_deriv_nonneg
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_dyn : ∀ t ≥ 0,
        psi1_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)
        + alpha2 * psi1 δ (traj t) d_des ≥ 0) :
    ∀ t ≥ 0,
      0 ≤ Real.exp (alpha2 * t) *
            (psi1_derivative (traj t) d_des
              (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)
              + alpha2 * psi1 δ (traj t) d_des) := by
  intro t ht
  have h_exp_pos : 0 < Real.exp (alpha2 * t) := by positivity
  exact mul_nonneg (le_of_lt h_exp_pos) (h_dyn t ht)

lemma psi1_scaled_monotone
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_dyn : ∀ t ≥ 0,
        psi1_derivative (traj t) d_des
          (safe_control_input δ (traj t) d_des (a_L t) nominal_a_i)
        + alpha2 * psi1 δ (traj t) d_des ≥ 0) :
    MonotoneOn (psi1_scaled δ traj d_des) (Set.Ici 0) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
  · intro x hx
    exact (psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj
          x hx).differentiableAt.continuousAt.continuousWithinAt
  · intro x hx
    exact (psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj x
      (le_of_lt (by simpa [interior_Ici] using hx))).differentiableAt.differentiableWithinAt
  · intro x hx
    have hx0 : 0 < x := by simpa [interior_Ici] using hx
    have h := psi1_scaled_hasDerivAt δ d_des a_L traj nominal_a_i h_traj
                x (le_of_lt hx0)
    have hnn := psi1_scaled_deriv_nonneg δ d_des traj nominal_a_i a_L h_dyn
                x (le_of_lt hx0)
    rw [h.deriv]
    exact hnn

lemma psi1_forward_invariant
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ)
    (h_mono : MonotoneOn (psi1_scaled δ traj d_des) (Set.Ici 0))
    (h_init : psi1 δ (traj 0) d_des ≥ 0) :
    ∀ t ≥ 0, psi1 δ (traj t) d_des ≥ 0 := by
  intro t ht
  have h_mono_t :
      psi1_scaled δ traj d_des 0 ≤ psi1_scaled δ traj d_des t := by
    apply h_mono
    · simp
    · exact ht
    · exact ht
  have h_init_scaled : 0 ≤ psi1_scaled δ traj d_des 0 := by
    rw [psi1_scaled]
    norm_num
    exact h_init
  have h_scaled_nonneg : 0 ≤ psi1_scaled δ traj d_des t :=
    le_trans h_init_scaled h_mono_t
  have h_exp_pos : 0 < Real.exp (alpha2 * t) := by positivity
  have h_exp_nonzero : Real.exp (alpha2 * t) ≠ 0 := ne_of_gt h_exp_pos
  have h_scaled_nonneg_exp :
      0 ≤ Real.exp (alpha2 * t) * psi1 δ (traj t) d_des := by
    simpa [psi1_scaled] using h_scaled_nonneg
  have h_div :
      0 ≤ (Real.exp (alpha2 * t) * psi1 δ (traj t) d_des)
            / Real.exp (alpha2 * t) :=
    div_nonneg h_scaled_nonneg_exp (le_of_lt h_exp_pos)
  simpa [h_exp_nonzero] using h_div

lemma psi0_dynamics_nonneg
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_psi1 : ∀ t ≥ 0, psi1 δ (traj t) d_des ≥ 0) :
    ∀ t ≥ 0,
      HasDerivAt (fun τ => psi0 δ (traj τ) d_des)
        (barrier_derivative (traj t) d_des) t
      ∧
      barrier_derivative (traj t) d_des
        + alpha1 * psi0 δ (traj t) d_des ≥ 0 := by
  intro t ht
  constructor
  · exact psi0_deriv_along_traj δ d_des traj a_L nominal_a_i h_traj t
  · simpa [psi1, psi0] using h_psi1 t ht

lemma psi0_scaled_deriv_nonneg
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (h_dyn : ∀ t ≥ 0,
        barrier_derivative (traj t) d_des
        + alpha1 * psi0 δ (traj t) d_des ≥ 0) :
    ∀ t ≥ 0,
      0 ≤ Real.exp (alpha1 * t) *
            (barrier_derivative (traj t) d_des
              + alpha1 * psi0 δ (traj t) d_des) := by
  intro t ht
  have h_exp_pos : 0 < Real.exp (alpha1 * t) := by positivity
  exact mul_nonneg (le_of_lt h_exp_pos) (h_dyn t ht)

lemma psi0_scaled_monotone
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState) (nominal_a_i : ℝ)
    (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_dyn : ∀ t ≥ 0,
        barrier_derivative (traj t) d_des
        + alpha1 * psi0 δ (traj t) d_des ≥ 0) :
    MonotoneOn (psi0_scaled δ traj d_des) (Set.Ici 0) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
  · intro x hx
    exact (psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj
          x hx).differentiableAt.continuousAt.continuousWithinAt
  · intro x hx
    exact (psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj x
      (le_of_lt (by simpa [interior_Ici] using hx))).differentiableAt.differentiableWithinAt
  · intro x hx
    have hx0 : 0 ≤ x := le_of_lt (by simpa [interior_Ici] using hx)
    have h := psi0_scaled_hasDerivAt δ d_des traj a_L nominal_a_i h_traj x hx0
    have hnn := psi0_scaled_deriv_nonneg δ d_des traj h_dyn x hx0
    rw [h.deriv]
    exact hnn

lemma psi0_forward_invariant
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ)
    (h_mono : MonotoneOn (psi0_scaled δ traj d_des) (Set.Ici 0))
    (h_init : psi0 δ (traj 0) d_des ≥ 0) :
    ∀ t ≥ 0, psi0 δ (traj t) d_des ≥ 0 := by
  intro t ht
  have h_mono_t :
      psi0_scaled δ traj d_des 0 ≤ psi0_scaled δ traj d_des t := by
    apply h_mono
    · simp
    · exact ht
    · exact ht
  have h_init_scaled : 0 ≤ psi0_scaled δ traj d_des 0 := by
    rw [psi0_scaled]
    norm_num
    exact h_init
  have h_scaled_nonneg : 0 ≤ psi0_scaled δ traj d_des t :=
    le_trans h_init_scaled h_mono_t
  have h_exp_pos : 0 < Real.exp (alpha1 * t) := by positivity
  have h_exp_nonzero : Real.exp (alpha1 * t) ≠ 0 := ne_of_gt h_exp_pos
  have h_scaled_nonneg_exp :
      0 ≤ Real.exp (alpha1 * t) * psi0 δ (traj t) d_des := by
    simpa [psi0_scaled] using h_scaled_nonneg
  have h_div :
      0 ≤ (Real.exp (alpha1 * t) * psi0 δ (traj t) d_des)
            / Real.exp (alpha1 * t) :=
    div_nonneg h_scaled_nonneg_exp (le_of_lt h_exp_pos)
  simpa [h_exp_nonzero] using h_div

lemma barrier_forward_invariant
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ)
    (h_psi0 : ∀ t ≥ 0, psi0 δ (traj t) d_des ≥ 0) :
    ∀ t ≥ 0, barrier δ (traj t) d_des ≥ 0 := by
  intro t ht
  simpa [psi0] using h_psi0 t ht

theorem fixed_hocbf_forward_invariance
    (δ : ℝ) (d_des : ℝ) (traj : ℝ → VehicleState)
    (nominal_a_i : ℝ) (a_L : ℝ → ℝ)
    (h_traj : is_closed_loop_trajectory δ d_des traj a_L nominal_a_i)
    (h_dyn : ∀ t ≥ 0,
        barrier_derivative (traj t) d_des
          + alpha1 * psi0 δ (traj t) d_des ≥ 0)
    (h_init : initial_hocbf_conditions δ (traj 0) d_des) :
    ∀ t ≥ 0, barrier δ (traj t) d_des ≥ 0 := by
  have h_psi0_mono :
      MonotoneOn (psi0_scaled δ traj d_des) (Set.Ici 0) :=
    psi0_scaled_monotone δ d_des traj nominal_a_i a_L h_traj h_dyn
  have h_psi0_forward :
      ∀ t ≥ 0, psi0 δ (traj t) d_des ≥ 0 :=
    fun t ht => psi0_forward_invariant δ traj d_des h_psi0_mono h_init.1 t ht
  exact barrier_forward_invariant δ traj d_des h_psi0_forward

theorem fixed_closed_loop_spacing_safety
    (δ : ℝ) (traj : ℝ → VehicleState) (d_des : ℝ)
    (h_barrier : ∀ t ≥ 0, barrier δ (traj t) d_des ≥ 0)
    (h_delta_nonneg : 0 ≤ δ) :
    ∀ t ≥ 0, |spacing_error (traj t) d_des| ≤ δ := by
  intro t ht
  have h_barrier_t := h_barrier t ht
  have h_abs_delta : |δ| = δ := abs_of_nonneg h_delta_nonneg
  have h_sq : (spacing_error (traj t) d_des) ^ 2 ≤ δ ^ 2 := by
    unfold barrier at h_barrier_t
    nlinarith
  have h_abs_bound : |spacing_error (traj t) d_des| ≤ |δ| :=
    sq_le_sq.mp h_sq
  simpa [h_abs_delta] using h_abs_bound


-- ## 15. End-to-end fixed-δ spacing safety


-- Main theorem.  Under the constant-δ HOCBF filter, any closed-loop
-- trajectory that starts in `{ψ₀ ≥ 0, ψ₁ ≥ 0}` and obeys the zero-error
-- velocity / gain conditions remains inside the tube `|e(t)| ≤ δ`.
--

theorem fixed_delta_spacing_safety
    (params : FixedDeltaParams)
    (d_des : ℝ)
    (traj : ℝ → VehicleState)
    (a_L : ℝ → ℝ)
    (nominal_a_i : ℝ)
    (h_traj : is_closed_loop_trajectory params.δ d_des traj a_L nominal_a_i)
    (h_vel : ∀ t ≥ 0, ZeroErrorVelocityBound params (traj t))
    (h_gain : ZeroErrorGainCondition params)
    (h_init : initial_hocbf_conditions params.δ (traj 0) d_des) :
    ∀ t ≥ 0, |spacing_error (traj t) d_des| ≤ params.δ := by
  -- 1. Instantaneous HOCBF inequality
  have h_hocbf : ∀ t ≥ 0,
      HOCBF_Condition params.δ (traj t) d_des
        (safe_control_input params.δ (traj t) d_des (a_L t) nominal_a_i) :=
    closed_loop_hocbf_along_trajectory params d_des traj a_L nominal_a_i h_vel h_gain
  -- 2. Highest-order differential inequality
  have h_psi1_dyn : ∀ t ≥ 0,
      psi1_derivative (traj t) d_des
        (safe_control_input params.δ (traj t) d_des (a_L t) nominal_a_i)
        + alpha2 * psi1 params.δ (traj t) d_des ≥ 0 := by
    intro t ht
    exact h_hocbf t ht
  -- 3. ψ₁ stays non-negative (exp-weight + monotonicity)
  have h_psi1_nonneg : ∀ t ≥ 0, psi1 params.δ (traj t) d_des ≥ 0 := by
    have h_mono :=
      psi1_scaled_monotone params.δ d_des traj nominal_a_i a_L h_traj h_psi1_dyn
    exact fun t ht =>
      psi1_forward_invariant params.δ traj d_des h_mono h_init.2 t ht
  -- 4. First-order inequality for ψ₀
  have h_psi0_dyn : ∀ t ≥ 0,
      barrier_derivative (traj t) d_des
        + alpha1 * psi0 params.δ (traj t) d_des ≥ 0 := by
    intro t ht
    exact (psi0_dynamics_nonneg params.δ d_des traj nominal_a_i a_L
      h_traj h_psi1_nonneg t ht).2
  -- 5. Barrier stays non-negative
  have h_barrier_nonneg : ∀ t ≥ 0, barrier params.δ (traj t) d_des ≥ 0 := by
    have h_mono :=
      psi0_scaled_monotone params.δ d_des traj nominal_a_i a_L h_traj h_psi0_dyn
    have h_psi0 :=
      fun t ht => psi0_forward_invariant params.δ traj d_des h_mono h_init.1 t ht
    exact barrier_forward_invariant params.δ traj d_des h_psi0
  -- 6. h ≥ 0 and δ > 0 ⇒ |e| ≤ δ
  have h_delta_nonneg : 0 ≤ params.δ := le_of_lt params.h_delta_pos
  exact fixed_closed_loop_spacing_safety params.δ traj d_des
    h_barrier_nonneg h_delta_nonneg

end
end DBCCProject
