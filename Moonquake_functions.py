import numpy as np
import obspy 
from obspy.core.utcdatetime import UTCDateTime
import scipy
import os
import math
from matplotlib import mlab
from matplotlib.colors import Normalize
from obspy.imaging.cm import obspy_sequential
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    

def kde2D(x, y, bandwidth, xbins=300j, ybins=300j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)



def despike_step1(y,m,kernel_size,threshold):
    med_filt_tr = scipy.signal.medfilt(abs(y), kernel_size=kernel_size)
    fivemedian = 5 * med_filt_tr
    #spikes = abs(y) > fivemedian
    spikes = abs((y - med_filt_tr)) > threshold
    y_out = y.copy() # So we don’t overwrite y
    for i in np.arange(len(spikes)):
        if int(spikes[i]) != 0: # If we have an spike in position i
            if i >= (len(spikes) - kernel_size):
                w = np.arange(i-m,len(spikes))
                w2 = w[spikes[w] == 0]
                y_out[i] = np.mean(y[w2])
        
            elif i <= kernel_size:
                w = np.arange(0,i+1+m)
                w2 = w[spikes[w] == 0]
                y_out[i] = np.mean(y[w2])
            
            else:
                w = np.arange(i-m,i+1+m) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                y_out[i] = np.mean(y[w2]) # and we average their values
    return y_out,fivemedian


def despike_step2(y,m,kernel_size):
    med_filt_tr = scipy.signal.medfilt(abs(y), kernel_size=kernel_size)
    fivemedian = (m * med_filt_tr)+0.1 
    spikes = abs(y) > fivemedian
    y_out = y.copy() # So we don’t overwrite y 
    # plt.plot(abs(y))
    # plt.plot(med_filt_tr)
    # plt.plot(m* med_filt_tr)
    # plt.show()
    for i in np.arange(len(spikes)):
        if int(spikes[i]) != 0: # If we have an spike in position i
            if i >= (len(spikes) - kernel_size):
                w = np.arange(i-m,len(spikes))
                w2 = w[spikes[w] == 0] 
                y_out[i] = np.mean(y[w2])
            
            elif i <= kernel_size:
                w = np.arange(0,i+1+m)
                w2 = w[spikes[w] == 0] 
                y_out[i] = np.mean(y[w2])
                
            else:
                w = np.arange(i-m,i+1+m) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                y_out[i] = np.mean(y[w2]) # and we average their values
    return y_out,fivemedian



def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b
    
def linear_interpolation(trace,interpolation_limit=1):
    """Snippet to interpolate missing data.  
    The SHZ traces have missing data samples 3-4 times every 32 samples. 
    Providing the seed data with these missing data would mean using very 
    large files. Instead, we provide the data with -1 replacing the gaps. 
    To change the files to interpolate across the gaps, use this simple method to 
    replace the -1 values. The trace is modified, and a mask is applied at 
    the end if necessary. 
    :type stream: :class:`~obspy.core.Trace` 
    :param trace: A data trace
    :type interpolation_limit: int 
    :param interpolation_limit: Limit for interpolation. Defaults to 1. For
      more information read the options for the `~pandas.Series.interpolate`
      method. 
    :return: original_mask :class:`~numpy.ndarray` or class:`~numpy.bool_`
       Returns the original mask, before any interpolation is made. 
    """

    trace.data = np.ma.masked_where(trace.data == -1, trace.data)
    original_mask = np.ma.getmask(trace.data)
    data_series = pd.Series(trace.data)
    # data_series.replace(-1.0, pd.NA, inplace=True)
    data_series.interpolate(method='linear', axis=0, limit=interpolation_limit, inplace=True, limit_direction=None, limit_area='inside')
    data_series.fillna(-1.0, inplace=True)
    trace.data=data_series.to_numpy(dtype=int)
    trace.data = np.ma.masked_where(trace.data == -1, trace.data)
    return original_mask

def maximize_plot(backend=None,fullscreen=False):
    """Maximize window independently of backend.
    Fullscreen sets fullscreen mode, that is same as maximized, but it doesn't have title bar (press key F to toggle full screen mode)."""
    if backend is None:
        backend=matplotlib.get_backend()
    mng = plt.get_current_fig_manager()

    if fullscreen:
        mng.full_screen_toggle()
    else:
        if backend == 'wxAgg':
            mng.frame.Maximize(True)
        elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
            mng.window.showMaximized()
        elif backend == 'TkAgg':
            mng.window.state('zoomed') #works fine on Windows!
        else:
            print ("Unrecognized backend: ",backend) #not tested on different backends (only Qt)
    plt.show()

    plt.pause(0.1) #this is needed to make sure following processing gets applied (e.g. tight_layout)
    


def spectrogram_a(data, samp_rate, per_lap=0.9, wlen=None, log=False,
                outfile=None, fmt=None, axes=None, dbscale=False,
                mult=8.0, cmap=obspy_sequential, zorder=None, title=None,
                show=True, clip=[0.0, 1.0]):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to a
        window length matching 128 samples.
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """
    import matplotlib.pyplot as plt
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)

    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
    return specgram,time,freq


