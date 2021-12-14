import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, pi, log10, exp
# from numpy.fft     import fft, ifft, fftshift  # Sometimes numpy is faster
from scipy.fftpack import fft, ifft, fftshift    # but usually scipy is faster
from scipy.special import factorial
from scipy.integrate import complex_ode
import scipy.ndimage
import time


def nlse(t, at, w0, gamma, betas, loss, fr=0.0, t1=0.0122, t2=0.032, flength=1, nsaves=200,
          atol=1e-4, rtol=1e-4, integrator='lsoda'):
    """
    This function propagates an optical input field (often a laser pulse)
    through a nonlinear material using the generalized nonlinear
    Schrodinger equation, which takes into account dispersion and
    nonlinearity. It is a "one dimensional" calculation, in that it doesn't
    capture things like self focusing and other geometric effects. It's most
    appropriate for analyzing light propagating through optical fibers.

    This code is based on the Matlab code found at www.scgbook.info,
    which is based on Eqs. (3.13), (3.16) and (3.17) of the book
    "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and
    J. R. Taylor (Cambridge 2010).
    The original Matlab code was written by J.C. Travers, M.H. Frosz and J.M.
    Dudley (2009). They ask that you please cite this chapter in any
    publication using this code.

    2018-02-01 - First Python port by Dan Hickstein (danhickstein@gmail.com)
    2020-01-11 - General clean up and PEP8

    Parameters
    ----------
    t : 1D numpy array of length n
        The time grid in picoseconds. Should be evenly spaced.
    at : 1D numpy array of length n
        The temporal pulse envelope. Matches the time grid T. Can be complex.
    w0 : float
        The "carrier frequency" for the moving reference frame
    gamma : float
        The effective nonlinearity, in units of [1/(W m)].
        note that the gammas for fibers are often described
        in units of 1/(W km) (per kilometer rather than per meter).
    betas : list of floats
        the coefficients of the dispersion expansion.
        Given as betas = [beta2, beta3, beta4, ...]
        In units of [ps^2/m, ps^3/m, ps^4/m ...]
        Note that this betas array is used to generate the b array.
        Those who want to include an aibitrary dispersion could hack this
        function and supply the b array. b is to so-called wavenumber
        (often written as k) and is equal to
        (refractive index)*2*pi/wavelength.
    loss : float
        Loss in dB/m (check units...)
    fr : float
        Frequency domain raman. More info needed.
    rt : numpy array
        The time domain Raman response. Matches the time grid T.
    flength : float
        the fiber length [meters]
    nsaves : int
        the number of equidistant grid points along the fiber to return
        the field. Note that the integrator usually takes finer steps than
        this, the nsaves parameters simply determines what is returned by this
        function
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

    Returns
    -------
    z : 1D numpy array of length nsaves
        an array of the z-coordinate along the fiber where the results are
        returned
    AT : 2D numpy array, with dimensions nsaves x n
        The time domain field at every step. Complex.
    AW : 2D numpy array, with dimensions nsaves x n
        The frequency domain field at every step. Complex.
    w : 1D numpy array of length n
        The frequency grid (not angular freq).
    """

    n = t.size        # number of time/frequency points
    dt = t[1] - t[0]  # time step
    v = 2 * pi * linspace(-0.5/dt, 0.5/dt, n)  # *angular* frequency grid
    alpha = log10(10**(loss/10.))              # attenuation coefficient

    b = np.zeros_like(v)
    for i in range(len(betas)):        # Taylor expansion of GVD
        b += betas[i]/factorial(i+2) * v**(i+2)

    lin_operator = 1j*b - alpha*0.5        # linear operator

    if np.nonzero(w0):          # if w0>0 then include shock
        gamma = gamma/w0
        w = v + w0              # for shock w is true freq
    else:
        w = 1                   # set w to 1 when no shock

    # Raman response:
    rt = (t1**2+t2**2)/t1/t2**2*np.exp(-t/t2)*np.sin(t/t1)
    rt[t < 0] = 0           # heaviside step function
    
    rw = n * ifft(fftshift(rt))      # frequency domain Raman

    # shift to fft space  -- Back to time domain, right?
    lin_operator = fftshift(lin_operator)
    w = fftshift(w)

    # define function to return the RHS of Eq. (3.13):
    def rhs(z, aw):
        at = fft(aw * exp(lin_operator*z))               # time domain field
        it = np.abs(at)**2                    # time domain intensity

        if rt.size == 1 or np.isclose(fr, 0):  # no Raman case
            m = ifft(at*it)                    # response function
        else:
            print('raman')
            rs = dt * fr * fft(ifft(it) * rw)     # Raman convolution
            m = ifft(at*((1-fr)*it + rs))         # response function

        r = 1j * gamma * w * m * exp(-lin_operator*z)  # full RHS of Eq. (3.13)
        return r

    z = linspace(0, flength, nsaves)    # select output z points

    aw = ifft(at.astype('complex128'))  # ensure integrator knows it's complex

    r = complex_ode(rhs).set_integrator(integrator, atol=atol, rtol=rtol)
    r.set_initial_value(aw, z[0])  # set up the integrator

    # intialize array for results:
    AW = np.zeros((z.size, aw.size), dtype='complex128')
    AW[0] = aw        # store initial pulse as first row

    start_time = time.time()  # start the timer

    for count, zi in enumerate(z[1:]):
        print('% 6.1f%% complete - %.1f seconds' % ((zi/z[-1])*100,
                                                    time.time()-start_time))
        if not r.successful():
            print('integrator failed!')
            break

        AW[count+1] = r.integrate(zi)

    # process the output:
    AT = np.zeros_like(AW)
    for i in range(len(z)):
        AW[i] = AW[i] * exp(lin_operator.transpose()*z[i])  # change variables
        AT[i, :] = fft(AW[i])            # time domain output
        AW[i, :] = fftshift(AW[i])  
        AW[i, :] = AW[i, :] * dt * n     # Original Dudley scaling factor
    
    return z, AT, AW, (v + w0)/(2*np.pi)


def test():
    """
    This function demonstrates how to call the gnlse function.
    This simulations demonstrates supercontinuum generation in an optical fiber
    using parameters similar to
    to Fig.3 of Dudley et. al, RMP 78 1135 (2006)
    """

    # simulation parameters:
    n = 2**13                   # number of grid points
    twidth = 12.5               # width of time window [ps]
    c = 299792458*1e9/1e12      # speed of light [nm/ps]
    wavelength = 835            # reference wavelength [nm]
    w0 = 2.0*pi*c/wavelength    # reference frequency [2*pi*THz]
    t = np.linspace(-twidth*0.5, twidth*0.5, n)  # time grid
    nsaves = 200                # number of length steps to save field at

    # input pulse parameters:
    power = 10000              # peak power of input [W]
    t0 = 0.0284                # duration of input [ps]
    at = np.sqrt(power)/np.cosh(t/t0)  # input field [W^(1/2)]
    
    # fiber parameters:
    flength = 0.15             # fibre length [m]
    flength = 0.001          # fibre length [m]


    # betas = [beta2, beta3, ...] in units [ps^2/m, ps^3/m ...]
    betas = [-11.830e-3, 8.1038e-5, -9.5205e-8, 2.0737e-10,
             -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21, -1.7140e-24]

    gamma = 0.11               # nonlinear coefficient [1/W/m]
    loss = 0                   # loss [dB/m]

    # propagate!
    z, AT, AW, w = nlse(t, at, w0, gamma, betas, loss=loss,
                         flength=flength, nsaves=nsaves)

    IW_dB = 10*log10(np.abs(AW)**2)  # log scale spectral intensity
    new_wls = np.linspace(400, 1350, 400)

    NEW_WLS, NEW_Z = np.meshgrid(new_wls, z)
    NEW_W = 2*pi*c/NEW_WLS

    # fast interpolation to wavelength grid,
    # so that we can plot using imshow for fast viewing:
    IW_WL = scipy.ndimage.interpolation.map_coordinates(
        np.abs(AW)**2, ((NEW_Z-np.min(z))/(z[1]-z[0]),
                        (NEW_W-np.min(w))/(w[1]-w[0])),
        order=1, mode='nearest')

    IW_dB = 10*np.log10(IW_WL)
    IT_dB = 10*np.log10(np.abs(AT)**2)

    fig, axs = plt.subplots(1, 2, figsize=(8, 5), tight_layout='True')

    axs[0].imshow(IW_dB, aspect='auto', origin='lower',
                  extent=(new_wls.min(), new_wls.max(), z.min(), z.max()),
                  clim=(-50, 0), cmap='jet')

    axs[1].imshow(IT_dB, aspect='auto', origin='lower',
                  extent=(t.min(), t.max(), z.min(), z.max()),
                  clim=(-50, 0), cmap='jet')

    axs[0].set_xlabel('Wavelength (nm)')
    axs[0].set_ylabel('Propagation length (meters)')

    axs[1].set_xlim(-0.5, 5)
    axs[1].set_xlabel('Time (ps)')

    plt.savefig('Dudley comparison.png', dpi=200)
    plt.show()
    



if __name__ == '__main__':
    test()
