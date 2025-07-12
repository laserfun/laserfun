⚡ laserfun ⚡
=============================
Some fun functions for modeling laser pulses in Python. 

Documentation is available at: `laserfun.readthedocs.io <https://laserfun.readthedocs.io/>`__

😊 Introduction
---------------

So far, laserfun consists mainly of:

- The Pulse class, which handles the amplitude of the electric field of the pulse in the time and frequency domains. 
- The Fiber class, which keeps track of the properties of the fiber, including the dispersion, nonlinearity, length, loss, etc. (By "fiber", we refer to any medium where it is appropriate to model a laser pulse in a single spatial mode which does not change over the propagation length. So, optical fibers and optical waveguides would be the most appropriate. Short distances of free-space propagation may also be appropriate, but important effects such as diffraction are ignored.)
- The NLSE function, which models the propagation of a pulse object through a fiber object according to the generalized nonlinear Schrodinger equation (GNLSE) as described in "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and J. R. Taylor (Cambridge 2010). The GNLSE propagates an optical input field (a laser pulse) through a nonlinear material (a fiber) and takes into account dispersion and Chi-3 nonlinearity.

🚀 Installation 🚀
------------------

Laserfun is a Python package, so you need Python to run it. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.anaconda.com/products/individual>`__, which is available for free.

Next, download this repository by clicking on the "Code" button and selecting "Download ZIP". Unzip the folder. Open a terminal or "Anaconda Prompt", navigate to the laserfun folder and type:

     ``pip install .``
     
If you would like to be able to edit the source code in the folder, install using:

     ``pip install -e .``
     
This method of installation allows you to modify the source code in-place without re-installing each time.


laserfun requires Python 3.9+. `NumPy <https://www.numpy.org/>`__ and `SciPy <https://www.scipy.org/>`__ (version 1.6.0+) are also required, and `Matplotlib <https://matplotlib.org/>`__ is required to run the examples. 



🤪 Example
----------

Here is a basic example that generates a pulse object using a 50-fs sech function, creates a fiber object with some dispersion, and propagates the pulse through the fiber using the NLSE. 

.. code-block:: python

    import laserfun as lf

    pulse = lf.Pulse(pulse_type='sech', fwhm_ps=0.050, epp=50e-12, center_wavelength_nm=1550)
    fiber1 = lf.Fiber(length=0.010, center_wl_nm=1550, dispersion=(-0.12, 0, 5e-6), gamma_W_m=1)
    results = lf.NLSE(pulse, fiber1)

    results.plot(units='dBm/THz')
    
Here is the output:

.. image:: https://user-images.githubusercontent.com/1107796/147493621-f4dee0aa-8618-47d0-9063-affd13543765.png
   :width: 600px
   :alt: example NLSE output

Additional examples are located in the `examples` directory. 


😎 Contributing
---------------
The laserfun project welcomes suggestions and pull request! The best place to start is to open a new Issue here: https://github.com/laserfun/laserfun/issues.

The following subsections contain a few notes for developers.

🐙 Unit tests
~~~~~~~~~~~~~
To run the tests, go to the PyNLSE folder and run:

    ``pytest nlse  -v  --cov=nlse``

Coverage can be checked with:

    ``coverage html``

which generates a html file that shows which lines are covered by the tests.


🐿️ Building documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation, go to the laserfun/doc folder on the command line and type:

    make html
    
Then, you can open 

    laserfun/doc/build/html/index.html
    
in a web browser to view the documentation. 


🍻 License
----------
laserfun is distributed under the MIT License. 

Enjoy!

