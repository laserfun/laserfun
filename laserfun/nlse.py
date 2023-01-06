"""Functions related to propagation of pulses according to the NLSE."""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, pi, exp, sin
from scipy.integrate import complex_ode
from scipy import constants
import scipy.ndimage
import time
from scipy.fftpack import fft, ifft, fftshift

# speed of light in m/s and nm/ps
c_mks = 299792458.0
c_nmps = c_mks * 1e9/1e12


def NLSE(pulse, fiber, nsaves=200, atol=1e-4, rtol=1e-4, reload_fiber=False,
         raman=False, shock=True, integrator='lsoda', print_status=True):
    """Propagate an laser pulse through a fiber according to the NLSE.

    This function propagates an optical input field (often a laser pulse)
    through a nonlinear material using the generalized nonlinear
    Schrodinger equation, which takes into account dispersion and
    nonlinearity. It is a "one dimensional" calculation, in that it doesn't
    capture things like self focusing and other geometric effects. It's most
    appropriate for analyzing light propagating through optical fibers.

    This code is based on the Matlab code found at www.scgbook.info,
    which is based on Eqs. (3.13), (3.16) and (3.17) of the book
    "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and
    J. R. Taylor (Cambridge 2010). The original Matlab code was written by
    J.C. Travers, M.H. Frosz and J.M. Dudley (2009). They ask that you cite
    the book in publications using their code.

    Parameters
    ----------
    pulse : pulse object
        This is the input pulse.
    fiber : fiber object
        This defines the media ("fiber") through which the pulse propagates.
    nsaves : int
        The number of equidistant grid points along the fiber to return
        the field. Note that the integrator usually takes finer steps than
        this, the nsaves parameters simply determines what is returned by this
        function.
    atol : float
        Absolute tolerance for the integrator. Smaller values produce more
        accurate results but require longer integration times. 1e-4 works well.
    rtol : float
        Relative tolerance for the integrator. 1e-4 work well.
    reload_fiber : boolean
        This determines if the fiber information is reloaded at each step. This
        should be set to True if the fiber properties (gamma, dispersion) vary
        as a function of length.
    raman : boolean
        Determines if the Raman effect will be included. Default is False.
    shock : boolean
        Determines if the self-steepening (shock) term will be taken into
        account. This is especially important for situations where the
        slowly varying envelope approximation starts to break down, which
        can occur for large bandwidths (short pulses).
    integrator : string
        Selects the integrator that will be passes to scipy.integrate.ode.
        options are 'lsoda' (default), 'vode', 'dopri5', 'dopri853'.
        'lsoda' is a good option, and seemed fastest in early tests.
        I think 'dopri5' and 'dopri853' are simpler Runge-Kutta methods,
        and they seem to take longer for the same result.
        'vode' didn't seem to produce good results with "method='adams'", but
        things werereasonable with "method='bdf'"
        For more information, see:
        docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    print_status : boolean
         This determines if the propagation status will be printed. Default
         is True.

    Returns
    -------
    results : PulseData object
        This object contains all of the results. Use
        ``z, f, t, AW, AT = results.get_results()``
        to unpack the z-coordinates, frequency grid, time grid, amplitude at
        each z-position in the freuqency domain, and amplitude at each
        z-position in the time domain.
    """
    # get the pulse info from the pulse object:
    t = pulse.t_ps  # time array in picoseconds
    at = pulse.at   # amplitude for those times in sqrt(W)
    w0 = pulse.centerfrequency_THz * 2 * pi  # center freq (angular!)

    n = t.size        # number of time/frequency points
    dt = t[1] - t[0]  # time step
    v = 2 * pi * linspace(-0.5/dt, 0.5/dt, n)  # *angular* frequency grid

    flength = fiber.length  # get length of fiber

    def load_fiber(fiber, z=0):
        # gets the fiber info from the fiber object
        gamma = fiber.get_gamma(z)  # gamma should be in 1/(W m), not 1/(W km)
        b = fiber.get_B(pulse, z)

        loss = np.log(10**(fiber.get_alpha(z)*0.1))  # convert from dB/m

        lin_operator = 1j*b - loss*0.5  # linear operator

        if w0 > 0 and shock:          # if w0>0 then include shock
            gamma = gamma/w0
            w = v + w0              # for shock w is true freq
        else:
            w = 1 + v*0             # set w to 1 when no shock

        # some fft shifts to things line up later:
        lin_operator = fftshift(lin_operator)
        w = fftshift(w)
        return lin_operator, w, gamma

    lin_operator, w, gamma = load_fiber(fiber)  # load fiber info

    # Raman response:
    if raman == 'dudley' or raman:
        fr = 0.18
        t1 = 0.0122
        t2 = 0.032
        rt = (t1**2+t2**2)/t1/t2**2*exp(-t/t2)*sin(t/t1)
        rt[t < 0] = 0                # heaviside step function
        rw = n * ifft(fftshift(rt))  # frequency domain Raman
    elif not raman:
        fr = 0
    else:
        raise ValueError('Raman method not supported')

    # define function to return the RHS of Eq. (3.13):
    def rhs(z, aw):
        nonlocal lin_operator, w, gamma

        if reload_fiber:
            lin_operator, w, gamma = load_fiber(fiber, z)

        at = fft(aw * exp(lin_operator*z))    # time domain field
        it = np.abs(at)**2                    # time domain intensity

        if np.isclose(fr, 0):  # no Raman case
            m = ifft(at*it)                    # response function
        else:
            rs = dt * fr * fft(ifft(it) * rw)     # Raman convolution
            m = ifft(at*((1-fr)*it + rs))         # response function

        r = 1j * gamma * w * m * exp(-lin_operator*z)  # full RHS of Eq. (3.13)
        return r

    z = linspace(0, flength, nsaves)    # select output z points
    aw = ifft(at.astype('complex128'))  # ensure integrator knows it's complex

    # set up the integrator:
    r = complex_ode(rhs).set_integrator(integrator, atol=atol, rtol=rtol)
    r.set_initial_value(aw, z[0])

    # intialize array for results:
    AW = np.zeros((z.size, aw.size), dtype='complex128')
    AW[0] = aw        # store initial pulse as first row

    start_time = time.time()  # start the timer

    for count, zi in enumerate(z[1:]):

        if print_status:
            print('% 6.1f%% - %.3e m - %.1f seconds' % ((zi/z[-1])*100,
                  zi, time.time()-start_time))
        if not r.successful():
            raise Exception('Integrator failed! Check the input parameters.')

        AW[count+1] = r.integrate(zi)

    # process the output:
    AT = np.zeros_like(AW)
    for i in range(len(z)):
        AW[i] = AW[i] * exp(lin_operator.transpose()*z[i])  # change variables
        AT[i, :] = fft(AW[i])            # time domain output
        AW[i, :] = fftshift(AW[i])

        # Below is the original dudley scaling factor that I think gives units
        # of sqrt(J/Hz) for the AW array. Removing this gives units that agree
        # with PyNLO, that seem to be sqrt(J*Hz) = sqrt(Watts) -DH 2021-12-15
        # AW[i, :] = AW[i, :] * dt * n

    pulse_out = pulse.create_cloned_pulse()
    pulse_out.at = AT[-1]

    results = PulseData(z, AW, AT, pulse, pulse_out, fiber)

    return results


class PulseData:
    """Process data from a pulse propagation.

    The following list of parameters is the attributes of the class.

    Parameters
    ----------
    z : array of length n
        The z-values corresponding the the propagation.
    f_THz : array of length m
        The values of the frequency grid in THz.
    t_ps : array of length m
        The values of time grid in ps
    AT : 2D array
        The complex amplitude of the electric field in the time domain at each
        position in z.
    AW : 2D array
        The complex amplitude of the electric field in the freq. domain at
        each position in z.
    pulse_in : pulse object
        The input pulse object.
    pulse_out : pulse object
        The output pulse object.
    fiber : fiber object
        The fiber object that the pulse was propagated through.
    """

    def __init__(self, z, AW, AT, pulse_in, pulse_out, fiber):
        self.z = z
        self.AW = AW
        self.AT = AT
        self.pulse_in = pulse_in
        self.pulse_out = pulse_out
        self.fiber = fiber
        self.f_THz = pulse_out.f_THz
        self.t_ps = pulse_out.t_ps

    def get_results(self, data_type='amplitude', rep_rate=1):
        """Get the frequency domain (AW) and time domain (AT) results of the
        NLSE propagation. Also provides the length (z), frequnecy (f), and time
        (t) arrays.

        ``'amplitude'`` - Native units for the NLSE, AT and AW are sqrt(W).
        Does NOT consider the rep-rate.

        ``'intensity'`` - Absolute value of amplitude squared. These units make
        some sense for AT, since they are J/sec, so integrating over the pulse
        works as expected. Units for AW are J*Hz, so be careful when
        integrating. Does NOT consider rep-rate.

        ``'mW/bin'`` - AW and AT are in mW per bin, so naive summing provides
        average power (in mW). Rep-rate taken into account.

        ``'mW/THz'`` - returns AW in units of mW/THz and AT in mW/ps. Rep-rate
        taken into account.

        ``'dBm/THz'`` - returns AW in units of mW/THz and AT in dBm/ps.
        Rep-rate taken into account.

        ``'mW/nm'`` - returns AW in units of mW/nm and AT in mW/ps. Rep-rate
        taken into account.

        ``'dBm/nm'`` - returns AW in units of dBm/nm and AT in dBm/ps. Rep-rate
        taken into account.

        In the above, dBm is 10*log10(mW).

        Note that, for the "per nm" situations, AW is still
        returned on a grid of even *frequencies*, so the uneven wavelength
        spacing should be taken into account when integrating. Use
        ``get_results_wavelength`` to re-interpolate to an evenly spaced
        wavelength grid.

        In order to get per-pulse numbers for all methods, simple set the rep-
        rate to 1.


        Parameters
        ----------
        data_type : 'string'
            Determines the units for the returned AW and AT arrays.

        rep_rate : float
            The repetition rate of the pulses for calculation of average power
            units. Does not affect the "amplitude" or "intensity" calculations,
            but scales all other calculations.

        Returns
        -------
        z : 1D numpy array of length nsaves
            Array of the z-coordinate along fiber, units of meters.
        f_THz : 1D numpy array of length n.
            The frequency grid (not angular freq) in THz.
        t_ps : 1D numpy array of length n
            The temporal grid in ps.
        AW : 2D numpy array, with dimensions nsaves x n
            The complex amplitide of the frequency domain field at every step.
        AT : 2D numpy array, with dimensions nsaves x n
            The complex amplitude of the time domain field at every step.
        """

        if data_type == 'amplitude':
            AW, AT = self.AW, self.AT

        elif data_type == 'intensity':
            AW = np.abs(self.AW)**2
            AT = np.abs(self.AT)**2

        elif data_type == 'dB':
            AW = dB(self.AW)
            AT = dB(self.AT)

        elif (data_type == 'mW/bin' or data_type == 'mW/THz' or
              data_type == 'dBm/THz' or data_type == 'mW/nm' or
              data_type == 'dBm/nm'):

            z = self.z
            f = self.pulse_in.f_THz
            df = (f[1]-f[0]) * 1e12  # df in Hz

            # per bin units:
            J_Hz = np.abs(self.AW)**2
            J_per_bin = J_Hz / df  # go from J*Hz/bin (native units) to J/bin
            # multiply by rep rate to get W/bin, and then mW/bin:
            mW_per_bin = J_per_bin * rep_rate * 1e3

            # per THz units:
            mW_per_THz = mW_per_bin / (df * 1e-12)
            dBm_per_THz = 10 * np.log10(mW_per_THz)

            # per wavelength units
            wl_nm = c_nmps / f
            wl_m = wl_nm * 1e-9
            nm_per_bin = wl_m**2 / c_mks * df * 1e9  # Jacobian from THz to nm
            NM_PER_BIN, Z = np.meshgrid(nm_per_bin, z)
            mW_per_nm = mW_per_bin / NM_PER_BIN    # convert to mW/nm
            dBm_per_nm = 10 * np.log10(mW_per_nm)  # convert to dBm/nm

            # AT conversion
            t = self.pulse_in.t_ps
            dt = (t[1] - t[0]) * 1e-12
            # convert from J/sec to J by mutiplying by dt
            AT_J_per_bin = np.abs(self.AT)**2 * dt
            # convert J/bin to mW/bin by multiplying by rep rate * 1e3
            AT_mW_per_bin = AT_J_per_bin * rep_rate * 1e3
            AT_mW_per_ps = AT_mW_per_bin / (dt * 1e12)
            AT_dBm_per_ps = 10*np.log10(AT_mW_per_ps)

            if data_type == 'mW/bin':
                AW = mW_per_bin
                AT = AT_mW_per_bin
            elif data_type == 'mW/THz':
                AW = mW_per_THz
                AT = AT_mW_per_ps
            elif data_type == 'dBm/THz':
                AW = dBm_per_THz
                AT = AT_dBm_per_ps
            elif data_type == 'mW/nm':
                AW = mW_per_nm
                AT = AT_mW_per_ps
            elif data_type == 'dBm/nm':
                AW = dBm_per_nm
                AT = AT_dBm_per_ps
            else:
                raise ValueError('Units not recognized.')

        else:
            raise ValueError('data_type not recognized.')

        return self.z, self.f_THz, self.t_ps, AW, AT

    def get_results_wavelength(self, wmin=None, wmax=None, wn=None,
                               data_type='intensity', rep_rate=1):
        """Get results on a wavelength grid.

        Re-interpolates the AW array from evenly-spaced frequencies to
        evenly-spaced wavelengths.

        Parameters
        ----------
        wmin : float or None
            the minimum wavelength for the re-interpolation grid.
            If None, it defaults to 0.25x the center wavelength of the pulse.
            If a float, this is the minimum wavelength of the pulse in nm
        wmax : float or None
            the minimum wavelength for the re-interpolation grid.
            If None, it defaults to 4x the center wavelength of the pulse.
            If a float, this is the maximum wavelength of the pulse in nm
        wn : int or None
            number of wavelengths for the re-interpolation grid.
            If None, it defaults to the number of points in AW multiplied by 2
            If an int, then this is just the number of points.
        data_type : 'string'
            Determines the units for the AW and AT arrays. See the
            documentation for the ``get_results`` function for more
            information. Note that ``data_type='amplitude'`` is supported but
            not recommended because interpolation on the rapidly varying grid
            of complex values can lead to inconsistent results.
        rep_rate : float
            The repetition rate of the pulses for calculation of average power
            units. Does not affect the "amplitude" or "intensity" calculations,
            but scales all other calculations.

        Returns
        -------
        z : 1D numpy array of length nsaves
            Array of the z-coordinate along fiber.
        wls : 1D numpy array of length n.
            The wavelength grid.
        t : 1D numpy array of length n
            The temporal grid.
        AW_WLS : 2D numpy array, with dimensions nsaves x n
            The complex amplitide of the frequency domain field at every step.
        AT : 2D numpy array, with dimensions nsaves x n
            The complex amplitude of the time domain field at every step.
        """
        z, f, t, AW, AT = self.get_results(data_type=data_type, 
                                           rep_rate=rep_rate)

        c_nmps = constants.value('speed of light in vacuum')*1e9/1e12

        if wmin is None:
            wmin = 0.25 * c_nmps/self.pulse_in.centerfrequency_THz
        if wmax is None:
            wmax = 4.0 * c_nmps/self.pulse_in.centerfrequency_THz
        if wn is None:
            wn = AW.shape[1] * 2

        new_wls = np.linspace(wmin, wmax, wn)

        NEW_WLS, NEW_Z = np.meshgrid(new_wls, z)
        NEW_F = c_nmps/NEW_WLS

        # fast interpolation to wavelength grid, so that we can plot using
        # imshow for fast viewing. This requires Scipy > 1.6.0.
        AW_WLS = scipy.ndimage.interpolation.map_coordinates(
            AW, ((NEW_Z-np.min(z))/(z[1]-z[0]),
                 (NEW_F-np.min(f))/(f[1]-f[0])),
            order=1, mode='nearest')

        return z, new_wls, t, AW_WLS, AT

    def plot(self, flim=30, tlim=50, margin=0.2, wavelength=False, show=True,
             units='intensity', rep_rate=1e8):
        """Plot the results in both the time and frequency domain.

        parameters
        ----------
        flim : float or array of length 2
            This sets the xlimits of the frequency domain plot. If this is a
            single number, it defines the dB level at which the plot will be
            set (with a margin). If an array of two values, this manually sets
            the xlims.
        tlim : float or array of length 2
            Same as flim, but for the time domain.
        margin : float
            Fraction to pad the xlimits. Default is 0.2.
        wavelength : boolean
            Determines if the "frequency" domain will be displayed in Frequency
            (THz) or wavelength (nm).
        show : boolean
            determines if plt.show() will be called to show the plot
        units : string
            Units for the frequency-domain plots. For example, dBm/THz mW/THz.
            See the documentation for the ``data_type`` keyword argument for
            the ``get_results`` method for more information.
        rep_rate : float
             The repetition rate of the pule train in Hz. This is used to
             calculate the average powers when using units other than
             "intensity".

        Returns
        -------
        fig : matplotlib.figure object
            The figure object so that modifications can be made.
        axs : an 2x2 array of axes objects
            The axes objects, so that modifications can be made.
            For example: axs[0, 1].set_xlim(0, 1000)
        """

        fig = plt.figure(figsize=(8, 8))
        ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
        ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
        ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
        ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

        z = self.z * 1e3  # convert to mm

        if units == 'amplitude':
            raise ValueError('Cannot plot amplitude.',
                             'Use intensity or other units.')
        elif units == 'intensity':
            funits = 'J * Hz'
            tunits = 'J/sec'
        elif units == 'mW/bin':
            funits = 'mW/bin'
            tunits = 'mW/bin'
        elif units == 'mW/THz':
            funits = 'mW/THz'
            tunits = 'mW/sec'
        elif units == 'dBm/THz':
            funits = 'dBm/THz'
            tunits = 'dBm/sec'
        elif units == 'mW/nm':
            funits = 'mW/nm'
            tunits = 'mW/sec'
        elif units == 'dBm/nm':
            funits = 'dBm/nm'
            tunits = 'dBm/sec'
        else:
            raise ValueError('Units not recognized.')

        if wavelength:
            ax0.set_xlabel('Wavelength (nm)')
            ax2.set_xlabel('Wavelength (nm)')
            q, f, t, IW, IT = self.get_results_wavelength(data_type=units,
                                                          rep_rate=rep_rate)

        else:
            ax0.set_xlabel('Frequency (THz)')
            ax2.set_xlabel('Frequency (THz)')
            junkz, f, t, IW, IT = self.get_results(data_type=units,
                                                   rep_rate=rep_rate)

        ax0.plot(f, IW[0], color='b', label='Initial')
        ax1.plot(t, IT[0], color='b', label='Initial')

        ax0.plot(f, IW[-1], color='r', label='Final')
        ax1.plot(t, IT[-1], color='r', label='Final')

        ax1.legend(loc='upper left', fontsize=9)

        ax1.set_xlabel('Time (ps)')

        ax0.set_ylabel('Intensity (%s)' % funits)
        ax1.set_ylabel('Intensity (%s)' % tunits)

        # when plotting in dB units, the plots look best if we set cmin to the
        # max value minus about 40 to 80 dB:
        if 'dB' in units:
            chif = np.max(IW)
            clof = np.max(IW) - 50
            chit = np.max(IT)
            clot = np.max(IT) - 80
            ylof = clof - 10
            yhif = chif + 10
            ylot = clot - 10
            yhit = chit + 10

            ax0.set_ylim(ylof, yhif)
            ax1.set_ylim(ylot, yhit)

        else:
            chif = np.max(IW)
            clof = np.min(IW)
            chit = np.max(IT)
            clot = np.min(IT)

        ax2.set_ylabel('Propagation distance (mm)')

        extf = (np.min(f), np.max(f), np.min(z), np.max(z))
        extt = (np.min(t), np.max(t), np.min(z), np.max(z))

        # TODO: figure out how to make the clims reasonable. I guess full-scale
        # for the linear units and max -40 or 80 for the other methods?

        ax2.imshow(IW, extent=extf, clim=(clof, chif), aspect='auto',
                   origin='lower')
        ax3.imshow(IT, extent=extt, clim=(clot, chit), aspect='auto',
                   origin='lower')

        ax3.set_xlabel('Time (ps)')

        fig.tight_layout()

        # set xlims:
        def find_width_and_center(x, y, offset=10):
            def find_roots(x, y):
                s = np.abs(np.diff(np.sign(y))).astype(bool)
                return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

            try:
                roots = find_roots(x, y - np.max(y) + offset)
                width = np.max(roots) - np.min(roots)
                center = (np.max(roots) + np.min(roots)) * 0.5

            except:
                width = np.max(x) - np.min(x)
                center = (np.max(x) + np.min(x)) * 0.5

            return width, center

        if not hasattr(flim, "__len__"):
            w, c = find_width_and_center(f, IW[-1], flim)
            flim = (c - 0.5*w*(1 + margin), c + 0.5*w*(1 + margin))

        if not hasattr(tlim, "__len__"):
            w, c = find_width_and_center(t, IT[-1], tlim)
            tlim = (c - 0.5*w*(1 + margin), c + 0.5*w*(1 + margin))

        ax2.set_xlim(flim[0], flim[1])
        ax3.set_xlim(tlim[0], tlim[1])
        
        for ax in (ax0, ax1):
            ax.grid(alpha=0.1, color='k')

        if show:
            plt.show()

        axs = np.array([[ax0, ax1], [ax2, ax3]])
        return fig, axs

    def calc_coherence(self, pulse_in, fiber, num_trials=5, n_steps=100,
                       random_seed=None,
                       noise_type='one_photon_freq', **nlse_kwargs):
        """
        This function runs several nlse simulations (given by num_trials), each
        time adding random noise to the pulse. By comparing the electric fields
        of the different pulses, an estimate of the coherence can be made.

        Parameters
        ----------
        pulse_in : pulse object

        num_trials : int
            this determines the number of trials to be run.

        random_seed : int
            this is the seed for the random noise generation. Default is None,
            which does not set a seed for the random number generator, which
            means that the numbers will be completely randomized.
            Setting the seed to a number (i.e., random_seed=0) will still
            generate random numbers for each trial, but the results from
            calculate_coherence will be completely repeatable.

        noise_type : str
            this specifies the method for including random noise onto the
            pulse. See :func:`pynlo.light.PulseBase.Pulse.add_noise` for the
            different methods.

        Returns
        -------
        g12W : 2D numpy array
            This 2D array gives the g12 parameter as a function of propagation
            distance and the frequency. g12 gives a measure of the coherence of
            the pulse by comparing several different trials.

        results : list of results for each trial
            This is a list, where each item of the list contains (z_positions,
            AW, AT, pulse_out), the results obtained from
            :func:`pynlo.interactions.FourWaveMixing.SSFM.propagate`.
        """

        results = []
        for num in range(0, num_trials):

            pulse = pulse_in.create_cloned_pulse()
            pulse.add_noise(noise_type=noise_type)

            y, AW, AT, pulse_out = self.propagate(
                pulse_in=pulse, fiber=fiber, n_steps=n_steps)

            results.append((y, AW, AT, pulse_in, pulse_out))

        for n1, (y, E1, AT, pulsein, pulseout) in enumerate(results):
            for n2, (y, E2, AT, pulsein, pulseout) in enumerate(results):
                if n1 == n2:
                    continue  # don't compare the same trial

                g12 = np.conj(E1)*E2/np.sqrt(np.abs(E1)**2 * np.abs(E2)**2)
                if 'g12_stack' not in locals():
                    g12_stack = g12
                else:
                    g12_stack = np.dstack((g12, g12_stack))

        # print g12_stack.shape, g12_stack.transpose().shape
        g12W = np.abs(np.mean(g12_stack, axis=2))

        return g12W, results


def dB(num):
    with np.errstate(divide='ignore'):
        return 10 * np.log10(np.abs(num)**2)
