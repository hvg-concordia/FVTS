import Mathlib

/-!
# TTC barrier certificate (Lean 4)

  s   = x_L − x_i − L
  Δv  = v_i − v_L
  TTC = s / Δv
  h   = TTC − τ
  𝒞   = { h ≥ 0 }          ⇔  TTC ≥ τ  ⇔  s ≥ τ Δv

Kinematics
  ṡ   = −Δv
  Δv̇  = a_i − a_L
  ḣ   = −1 − s (a_i − a_L) / Δv²

CBF (Ames et al., IEEE TAC 2017), α(h) = h
  ḣ + h ≥ 0
  ⇔  a_i ≤ U(a_L) := a_L + (h − 1) Δv² / s

Safe set is the original 𝒞. Existence of a legal a_i is the
predicate `Feasible`.

Default threshold in the original monitor: τ = 3 s.
-/

namespace TTCProject

set_option linter.style.longLine false
noncomputable section

structure VehicleState where
  x_L : ℝ
  v_L : ℝ
  x_i : ℝ
  v_i : ℝ
  L   : ℝ

structure VehicleInput where
  a_L : ℝ
  a_i : ℝ

def τ : ℝ := 3.0
def a_min : ℝ := -6
def a_max : ℝ := 3

lemma τ_pos : (0 : ℝ) < τ := by norm_num [τ]
lemma a_min_lt_a_max : a_min < a_max := by norm_num [a_min, a_max]
lemma a_min_le_a_max : a_min ≤ a_max := le_of_lt a_min_lt_a_max

def spacing (s : VehicleState) : ℝ := s.x_L - s.x_i - s.L
def relSpeed (s : VehicleState) : ℝ := s.v_i - s.v_L

def ValidInput (u : VehicleInput) : Prop :=
  a_min ≤ u.a_L ∧ u.a_L ≤ a_max ∧
  a_min ≤ u.a_i ∧ u.a_i ≤ a_max

def ValidState (s : VehicleState) : Prop :=
  0 < s.x_i ∧ 0 < s.x_L ∧
  0 < s.v_i ∧ 0 < s.v_L ∧
  0 < s.L ∧
  0 < spacing s ∧
  0 < relSpeed s

lemma spacing_pos (s : VehicleState) (h : ValidState s) : 0 < spacing s := h.2.2.2.2.2.1
lemma relSpeed_pos (s : VehicleState) (h : ValidState s) : 0 < relSpeed s := h.2.2.2.2.2.2
lemma relSpeed_sq_pos (s : VehicleState) (h : ValidState s) :
    0 < (relSpeed s) ^ 2 := sq_pos_of_pos (relSpeed_pos s h)

def TTC (s : VehicleState) : ℝ := spacing s / relSpeed s
def h (s : VehicleState) : ℝ := TTC s - τ
def SafeSet (s : VehicleState) : Prop := 0 ≤ h s

lemma safe_iff_ttc (s : VehicleState) : SafeSet s ↔ τ ≤ TTC s := by
  simp [SafeSet, h, sub_nonneg]

lemma safe_iff_gap (s : VehicleState) (hV : ValidState s) :
    SafeSet s ↔ τ * relSpeed s ≤ spacing s := by
  have hΔ := relSpeed_pos s hV
  constructor
  · intro hh
    exact (le_div_iff₀ hΔ).mp ((safe_iff_ttc s).mp hh)
  · intro hs
    exact (safe_iff_ttc s).mpr ((le_div_iff₀ hΔ).mpr hs)

def hdot (s : VehicleState) (u : VehicleInput) : ℝ :=
  - (1 : ℝ) - spacing s * (u.a_i - u.a_L) / (relSpeed s) ^ 2

lemma hdot_add_h (s : VehicleState) (u : VehicleInput) :
    hdot s u + h s =
      (h s - 1) - spacing s * (u.a_i - u.a_L) / (relSpeed s) ^ 2 := by
  unfold hdot; ring

def U (s : VehicleState) (a_L : ℝ) : ℝ :=
  a_L + (h s - 1) * (relSpeed s) ^ 2 / spacing s

def SafeA_i (s : VehicleState) (u : VehicleInput) : Prop :=
  u.a_i ≤ U s u.a_L

def Feasible (s : VehicleState) (a_L : ℝ) : Prop :=
  a_min ≤ U s a_L

theorem cbf_holds
    (s : VehicleState) (u : VehicleInput)
    (hV : ValidState s) (hSafe : SafeA_i s u) :
    0 ≤ hdot s u + h s := by
  have hs := spacing_pos s hV
  have hΔ2 := relSpeed_sq_pos s hV
  have hU : u.a_i ≤ u.a_L + (h s - 1) * (relSpeed s) ^ 2 / spacing s := by
    simpa [SafeA_i, U] using hSafe
  have hbound : u.a_i - u.a_L ≤ (h s - 1) * (relSpeed s) ^ 2 / spacing s := by
    linarith
  have hprod : spacing s * (u.a_i - u.a_L)
      ≤ (h s - 1) * (relSpeed s) ^ 2 := by
    have := mul_le_mul_of_nonneg_left hbound (le_of_lt hs)
    have hdiv : spacing s * ((h s - 1) * (relSpeed s) ^ 2 / spacing s)
        = (h s - 1) * (relSpeed s) ^ 2 := by
      field_simp [ne_of_gt hs]
    calc
      spacing s * (u.a_i - u.a_L)
          ≤ spacing s * ((h s - 1) * (relSpeed s) ^ 2 / spacing s) := this
      _ = (h s - 1) * (relSpeed s) ^ 2 := hdiv
  have hfrac : spacing s * (u.a_i - u.a_L) / (relSpeed s) ^ 2 ≤ h s - 1 := by
    have := div_le_div_of_nonneg_right hprod (le_of_lt hΔ2)
    have : (h s - 1) * (relSpeed s) ^ 2 / (relSpeed s) ^ 2 = h s - 1 := by
      apply (div_eq_iff (ne_of_gt hΔ2)).2
      ring
    rw [this] at ‹spacing s * (u.a_i - u.a_L) / (relSpeed s) ^ 2 ≤
      (h s - 1) * (relSpeed s) ^ 2 / (relSpeed s) ^ 2›
    assumption
  have hid := hdot_add_h s u
  linarith

def a_i_star (s : VehicleState) (a_L : ℝ) : ℝ :=
  max a_min (min a_max (U s a_L))

theorem a_i_star_ge_amin (s : VehicleState) (a_L : ℝ) :
    a_min ≤ a_i_star s a_L := le_max_left _ _

theorem a_i_star_le_amax (s : VehicleState) (a_L : ℝ) :
    a_i_star s a_L ≤ a_max := by
  unfold a_i_star
  apply max_le
  · exact a_min_le_a_max
  · exact min_le_left _ _

theorem a_i_star_le_U_of_feasible
    (s : VehicleState) {a_L : ℝ} (hF : Feasible s a_L) :
    a_i_star s a_L ≤ U s a_L := by
  unfold a_i_star Feasible at *
  apply max_le
  · exact hF
  · exact min_le_right _ _

theorem feasible_implies_witness
    (s : VehicleState) (a_L : ℝ) (hF : Feasible s a_L) :
    ∃ a_i : ℝ,
      a_min ≤ a_i ∧ a_i ≤ a_max ∧
      SafeA_i s ⟨a_L, a_i⟩ := by
  refine ⟨a_i_star s a_L, a_i_star_ge_amin s a_L, a_i_star_le_amax s a_L, ?_⟩
  exact a_i_star_le_U_of_feasible s hF

/-- Barrier certificate on the set 𝒞. -/
theorem barrier_certificate
    (s : VehicleState) (u : VehicleInput)
    (hV : ValidState s) (_hC : SafeSet s) (hSel : SafeA_i s u) :
    0 ≤ hdot s u + h s :=
  cbf_holds s u hV hSel

theorem hdot_le_neg_one_if_leader_at_amin
    (s : VehicleState) (u : VehicleInput)
    (hV : ValidState s) (hL : u.a_L = a_min) (hI : a_min ≤ u.a_i) :
    hdot s u ≤ -1 := by
  have hs := spacing_pos s hV
  have hΔ2 := relSpeed_sq_pos s hV
  have hdiff : 0 ≤ u.a_i - u.a_L := by simp [hL]; linarith
  have : 0 ≤ spacing s * (u.a_i - u.a_L) / (relSpeed s) ^ 2 :=
    div_nonneg (mul_nonneg (le_of_lt hs) hdiff) (le_of_lt hΔ2)
  unfold hdot; linarith

theorem not_feasible_on_boundary_worst
    (s : VehicleState) (hV : ValidState s) (hh : h s = 0) :
    ¬ Feasible s a_min := by
  have hs := spacing_pos s hV
  have hΔ2 := relSpeed_sq_pos s hV
  intro hF
  unfold Feasible U at hF
  simp [hh] at hF
  have : 0 < (relSpeed s) ^ 2 / spacing s := div_pos hΔ2 hs
  field_simp [ne_of_gt hs] at hF
  linarith


end
end TTCProject
