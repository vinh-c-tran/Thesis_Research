## Broyden's Method - N Dimensional Quasi Newton Root Finder Algorithm

For the purposes of this thesis project, we need to generate data that describes the composition of the interior of a neutron star. In particular, we would like to find the values of various particle fractions (equivalently fermi momenta), meson mediator field values, all as a function of baryon number density nB. 

In the framework of a Walecka-type relativistic mean field model, we have the mean field equations of motion. This together with additional constraints such as charge neutrality (the number of positively and negatively charged particles must be equal), baryon number conservation (nB = sum of all baryon densities), and beta equilibrium (a constraint on the chemical potentials), we end up with a system of N coupled, highly nonlinear equations and N unknowns. 

To solve this system, we use Broydens' method, a quasi-Newton method, for the root finder. In particular, our implementation here interfaces with Sympy. We use sympy to generate the system of equations which then get passed to this method. 

### Broyden.py 
This library contains functions that find the solution to a system of N non-linear equations. 

### Solver_helper.py
This library declares three particle classes: a baryon, lepton, and meson class which store both sympy (symbolic) attributes and numeric attributes such as coupling constants, momenta values, masses, etc. This then uses methods in Bryoden.py to generate a system of non-linear equations provided a list of particles, finds and solves the system for values of baryon density nB/n0 from 0.00 to 8.00 fm-3 and stores the output in a Pandas Dataframe. This Dataframe can then be manipulated or exported to a csv. Intended for use in a Jupyter notebook for best visualization of data. 

## References 
