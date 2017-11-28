
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift

import scipy

from skimage.util.shape import view_as_windows

import matplotlib.pyplot as plt 


# ==============================================================================
# Debug Plotting Functions
# ==============================================================================

PLOT_ON = False
SAVE_FIG = True
FIG_COUNT = 0
PLAY_AUDIO = True

"""
Takes the Fourier transform of samples, shifts them, and plots them
"""
def plot_fft(x,title='FFT',phaseplot=False,dbplot=True,normalize=False):
    X = fftshift(fft(x,n=4096))
    plot_freq(X,title,phaseplot,dbplot,normalize)

"""
Plots samples from a Fourier transform
"""
def plot_freq(X,title='FFT',phaseplot=False,dbplot=True,normalize=False):
    # checks if plotting has been disabled
    global PLOT_ON, SAVE_FIG, FIG_COUNT
    if not PLOT_ON:
        return

    # normalizes the fft
    if normalize:
        X /= len(X)

    # calculates xaxis for plot
    freq = np.arange(len(X)) / len(X) * 2 - 1
    
    # plots FFT
    if phaseplot:
        resp = np.angle(X)
        norm = resp/pi

        plt.plot(freq,norm)
        plt.ylabel('Normalized Phase')
    elif dbplot:
        resp = np.abs(X)
        norm = 20*np.log10(resp)

        plt.plot(freq,norm)
        plt.ylabel('Magnitude (dB)')
    else:
        resp = np.abs(X)

        plt.plot(freq,resp)
        plt.ylabel('Magnitude')

    plt.title(title)
    plt.xlabel('Normalized Freq')

    plt.grid()
    axes = plt.gca()
    axes.set_xlim([-1,1])

    if SAVE_FIG:
        fname = 'fig%02d.png' % FIG_COUNT
        plt.savefig(fname)
        print('Saved %s' % fname)
        FIG_COUNT += 1
        plt.gcf().clear()
    else:
        plt.show()

# ==============================================================================

################################################################################
################################################################################
################################################################################

# ==============================================================================

def time_alias(dat,n):
    assert len(dat) > n
    """
    if len(dat) < n:
        return dat
    """

    aliased = sum(view_as_windows(dat,n,n))

    overlen = len(dat) % n
    if overlen != 0:
        extra = np.zeros(n,dtype=aliased.dtype)
        extra[:overlen] = dat[-overlen:] 
        aliased += extra

    return aliased


"""
"""
def td_dft(dat, winlen, winoverlap, dftsize):
    views = view_as_windows(dat,winlen, winlen-winoverlap)

    if winlen > dftsize:
        views = [ time_alias(v,dftsize) for v in views ]

    dfts = [ fft(v,dftsize) for v in views ]

    return np.array(dfts)

# ==============================================================================

def heatplot(dat):
    plt.imshow(dat, cmap='hot', interpolation='nearest')
    plt.show()

# ==============================================================================

def spectrogram(dat, winlen, winoverlap, dftsize):
    mat = td_dft(dat, 256, 128, 512)
    mag = np.abs(mat)
    sqr = mag*mag
    #sqr = np.transpose(sqr)
    return sqr

def power_est(dat):
    None

def noise_est(spec, percent):
    sort = np.sort(spec.flatten())
    subset = sort[:int(len(sort)*percent)]
    avg = np.mean(subset)
    return avg

def band_occupancy(spec, sigma, eta):
    thresh = np.vectorize(lambda x: x >= sigma*eta)
    detects = thresh(spec)
    return detects

def strongest(spec, n=5):
    sort = np.sort(spec.flatten())
    biggest = sort[-n:]
    locs = [ np.where(spec == big) for big in biggest ]
    lst = [ (v,r[0],c[0]) for (v,(r,c)) in zip(biggest,locs) ]
    return lst

# ==============================================================================
# Processing Function
# ==============================================================================

def process(infile):
    # load data
    samples = scipy.fromfile(open(infile), dtype=scipy.complex64)

    # select how much data to process
    x = samples[:len(samples):8]

    spec = spectrogram(x,256,128,512)
    #heatplot(spec)

    power_est(x)

    sigma = noise_est(spec,.2)
    #print(sigma)

    detects = band_occupancy(spec, sigma, 20)
    #heatplot(detects)

    strong = strongest(spec)
    print(strong)

# ==============================================================================



# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    process('blind_test.raw')

# ==============================================================================

