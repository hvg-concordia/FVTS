# Lean 4 Formalizations

This directory contains Lean 4 formalizations for our work on barrier-based car-following safety. The developments cover TTC-, TTC–THW-, and spacing-based safety conditions and their associated control constraints.

## Files

- `TTC-BC.lean` formalizes the TTC-based barrier condition and its control constraints.
- `TTC-THW-CBF.lean` formalizes the combined TTC–THW condition and its control constraints.
- `HOCBF_FixedDelta.lean` formalizes the fixed-spacing HOCBF condition.
- `Adaptive_HOCBF.lean` formalizes the adaptive-spacing HOCBF condition.

## Setup 

1. Install VS Code and the [official Lean 4 extension](https://marketplace.visualstudio.com/items?itemName=leanprover.lean4). Follow its setup guide to install Lean.
2. Open a `.lean` file in VS Code and wait for Lean to process it.
3. Check the editor for errors and use the Lean Infoview to inspect proof goals.
4. If an import cannot be found, install the dependency that provides it.
