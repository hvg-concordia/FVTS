# Formally Verified Barrier Certificates for Car-Following Traffic Safety

We develop a formal approach to car-following safety that expresses safety requirements as barrier certificates and uses control barrier functions to translate them into constraints on vehicle acceleration. We formalize and verify the resulting mathematical properties in Lean 4, then use CARLA simulations to examine how the controllers behave in driving scenarios. 

## Contents

The repository includes:

- **Lean 4 developments** that formalize definitions, conditions, and proof results for TTC-based and TTC–THW-based safety barriers, as well as fixed and adaptive HOCBF spacing conditions.
- **CARLA scripts** that run or visualize longitudinal vehicle-interaction scenarios used to evaluate the corresponding safety controllers.

The scripts cover different experiments and may use different parameter values. Refer to the relevant source files for each experiment’s assumptions and settings.

## Running the code

Use the Lean 4 version and project configuration specified in the Lean directory. Use a compatible CARLA installation for the simulation scripts. Each script or experiment may have its own input files and configuration; consult its source and any accompanying instructions before running it.
