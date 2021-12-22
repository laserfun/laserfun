# pyNLSE
A simple generalized nonlinear Schrodinger equation (GNLSE) propagator in Python

This script consists of a function to integrate the generalized nonlinear Schrodinger equation (GNLSE) propagator written in Python. It's based on Eqs. (3.13), (3.16) and (3.17) of the book "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and J. R. Taylor (Cambridge 2010).

The integration is performed with the [scipy.integrat.complex_ode](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.complex_ode.html) function, which provides access to some fast and robust differential equation integrators, as discussed in the [scipy.integrate.ode](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode) documentation.

The GNLSE propagates an optical input field (often a laser pulse) through a nonlinear material and takes into account dispersion and Chi-3 nonlinearity. It is a "one dimensional" calculation and doesn't capture things like self focusing and other geometric effects. It's most appropriate for analyzing light propagating through optical fibers or waveguides and other situations where the mode of the light doesn't change as it propagates.

Unit tests
----------
To run the tests, go to the PyNLSE folder and run:

    pytest nlse  -v  --cov=nlse

Coverage can be checked with:

    coverage html

which generates a html file that shows which lines are covered by the tests.


Enjoy!

