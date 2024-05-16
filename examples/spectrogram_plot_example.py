"""Runs a simple example of NLSE pulse propagation and spectrogram plotting."""

import laserfun as lf
import numpy as np
import matplotlib.pyplot as plt

p = lf.Pulse(pulse_type='sech', fwhm_ps=0.05, epp=50e-12,
             center_wavelength_nm=1550, time_window_ps=7)

f = lf.Fiber(length=0.010, center_wl_nm=1550, dispersion=(-0.12, 0, 5e-6),
             gamma_W_m=1)

results = lf.NLSE(p, f, print_status=False)

pulse = results.pulse_out

if __name__ == '__main__':  # make plots if we're not running tests
    
    # Frequency plot
    results.plot(units='dBm/nm')
    fig, (ax0,ax1) = pulse.plot_spectrogram(wavelength_or_frequency='frequency',ylabels_of_interest = [100, 200,300])
    
    # Wavelength plot
    results.plot(units='dBm/nm',wavelength=True)
    fig, (ax0,ax1) = pulse.plot_spectrogram(wavelength_or_frequency='wavelength',ylabels_of_interest = [600,1200])

    # # Connect the click event handler
    from matplotlib.widgets import Button

    #### Event handler for click events
    def on_click(event):
        if event.inaxes == ax0:
            x = event.xdata
            ax0.axvline(x=x, color='red', linestyle='--')
            ax1.axvline(x=x, color='red', linestyle='--')
            ax0.text(x, ax0.get_ylim()[1], f'{x:.2f}', color='red', verticalalignment='top')
            ax1.text(x, ax1.get_ylim()[1], f'{x:.2f}', color='red', verticalalignment='top')
            fig.canvas.draw()

    # Toggle function for enabling/disabling the event handler
    def toggle_event_handler(event):
        if toggle_event_handler.cid is not None:
            fig.canvas.mpl_disconnect(toggle_event_handler.cid)
            toggle_event_handler.cid = None
            toggle_button.label.set_text("Enable Click")
        else:
            toggle_event_handler.cid = fig.canvas.mpl_connect('button_press_event', on_click)
            toggle_button.label.set_text("Disable Click")

    # Initialize the toggle_event_handler attribute
    toggle_event_handler.cid = None

    # Add a button to toggle the event handler
    ax_toggle = plt.axes([0, 0.97, 0.11, 0.03])  # Position: left, bottom, width, height
    toggle_button = Button(ax_toggle, 'Enable Click')
    toggle_button.on_clicked(toggle_event_handler)

