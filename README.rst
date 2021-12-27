laserfun README
===============

Some entertaining functions for modeling laser pulses in Python.

Introduction
------------

So far, ``laserfun`` consists mainly of:

- The Pulse class, which handles the amplitude of the electic field of the pulse in the time and frequency domains. 
- The Fiber class, which keeps track of the properties of the fiber, including the dipersion, nonlinearity, length, loss, etc. (By "fiber", we refer to any medium where it is appropriate to model a laser pulse in a single spatial mode which doesnt change over the propagation length. So, optical fibers and optical waveguides would be the most appropriate. Short distances of free-space propagation may also be appropriate, but important effects such as diffraction are ignored.)
- The NLSE function, which models the propagation of a pulse object through a fiber object according to the generalized nonlinear Schrodinger equation (GNLSE) as described in "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and J. R. Taylor (Cambridge 2010).The GNLSE propagates an optical input field (a laser pulse) through a nonlinear material (a fiber) and takes into account dispersion and Chi-3 nonlinearity.

Intallation
-----------

Requirements
~~~~~~~~~~~~

laserfun requires Python 3.9+. `NumPy <https://www.numpy.org/>`__ and `SciPy <https://www.scipy.org/>`__ (version 1.6.0+) are also required, and `Matplotlib <https://matplotlib.org/>`__ is required to run the examples. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.anaconda.com/products/individual>`__, which is available for free.

With pip
~~~~~~~~

Sorry, were not on PyPi just yet. But soon!

With setuptools
~~~~~~~~~~~~~~~

If you might contribute to the laserfun project, we recommend that you fork the repository to your own account, click "open in GitHub desktop", and save your fork somewhere on your computer. Then, navigate to the laserfun folder on the command line (Anaconda prompt in Windows or Terminal on the Mac) and type

    python setup.py develop

This method of installation allows you to modify the source code in-place without re-installing each time.

If you just want to install the code, then you can simply download this repository as a zip file, extract it, navigate to the laserfun folder on the command line, and type
    
        python setup.py install


Example of use
--------------

Here is a basic example that generates a pulse object using a 50-fs sech function, creates a fiber object with some dispersion, and propagates the pulse through the fiber using the NLSE. 

.. code-block:: python

    import laserfun as lf

    pulse = lf.Pulse(pulse_type='sech', fwhm_ps=0.050, center_wavelength_nm=1550,
                     time_window_ps=7, npts=2**12, epp=50e-12)

    fiber1 = lf.Fiber(0.010, center_wl_nm=1550, dispersion_format='GVD',
                      dispersion=(-0.12, 0, 5e-6), gamma_W_m=1)

    results = lf.NLSE(pulse, fiber1)

    results.plot()
    
Here is the output:

.. image:: https://user-images.githubusercontent.com/1107796/147493621-f4dee0aa-8618-47d0-9063-affd13543765.png
   :width: 600px
   :alt: example NLSE output

.. note:: Additional examples are located in the `examples` directory. 

Contributing
------------
The laserfun project welcomes suggestions and pull request! The best place to start is to open a new Issue here: https://github.com/laserfun/laserfun/issues.

The following subsections contain a few notes for developers.

Unit tests
~~~~~~~~~~
To run the tests, go to the PyNLSE folder and run:

    pytest nlse  -v  --cov=nlse

Coverage can be checked with:

    coverage html

which generates a html file that shows which lines are covered by the tests.

License
-------
laserfun is distributed under the MIT License. 

Enjoy!

