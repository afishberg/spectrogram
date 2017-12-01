
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift

import scipy

from skimage.util.shape import view_as_windows

from scipy.signal import hamming, blackman

import matplotlib.pyplot as plt 
#import matplotlib as mpl 
#import matplotlib.colors as colors

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

# ==============================================================================

def td_dft(dat, winlen, winoverlap, dftsize, winty='rect'):
    views = view_as_windows(dat,winlen, winlen-winoverlap)

    if winlen > dftsize:
        views = [ time_alias(v,dftsize) for v in views ]

    if winty == 'hamming':
        win = hamming(winlen)
    elif winty == 'blackman':
        win = blackman(winlen)
    else:
        win = np.array(winlen*[1])

    dfts = [ fftshift(fft(v*win,dftsize)) for v in views ]

    return np.array(dfts)

# ==============================================================================

def dB(x):
    return 10*np.log10(x)

# ==============================================================================

def heatplot(dat,smprate=2048000):
    """
    X, Y = np.mgrid[0:dat.shape[0]/smprate:complex(0,dat.shape[0]), -1:1:complex(0,dat.shape[-1])]

    Z1 = dat

    fig, ax = plt.subplots(2, 1)

    pcm = ax[0].pcolor(X, Y, Z1, cmap='PuBu_r')
    fig.colorbar(pcm, ax=ax[0], norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()), extend='max')

    pcm = ax[1].pcolor(X, Y, Z1, cmap='PuBu_r')
    fig.colorbar(pcm, ax=ax[1], extend='max')
    fig.show()
    """
    """
    X, Y = np.mgrid[0:dat.shape[0]/smprate:complex(0,dat.shape[0]), -1:1:complex(0,dat.shape[-1])]
    plt.imshow(X,Y,dat, interpolation='nearest')
    plt.show()
    """
    plt.imshow(dB(dat), cmap='hot', interpolation='nearest')
    plt.show()

# ==============================================================================

def spectrogram(dat, winlen, winoverlap, dftsize, winty='rect'):
    mat = td_dft(dat, winlen, winoverlap, dftsize, winty=winty)
    mag = np.abs(mat)
    sqr = mag*mag
    return sqr

def power_est(spec):
    pgram = np.sum(spec,axis=0) / spec.shape[0]
    return pgram

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

def process(samples,occ,eta):
    # winlen, winoverlap, dftsize
    spec = spectrogram(samples,4096,2048,4096,winty='blackman')
    heatplot(spec)

    pest = power_est(spec)

    sigma = noise_est(spec,occ) # .8

    plt.plot(dB(pest))
    plt.plot(np.array(spec.shape[-1]*[dB(sigma)]))
    plt.show()

    detects = band_occupancy(spec, sigma, eta)
    heatplot(detects)

    strong = strongest(spec)
    print(strong)

# ==============================================================================

def chirp_sample(a,dur,fs):
    t = np.arange(0,dur,1/fs)
    sig = np.cos( a*(t*t) )
    return sig

def ld_file(infile):
    return scipy.fromfile(open(infile), dtype=scipy.complex64)

# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    # data, percent noise estimate, eta
    #process(ld_file('blind_test.raw'), .5, 8)
    process(ld_file('blind_test_project02.raw'), .8, 7)
    #process(chirp_sample(pi,5,256000), .8, 3)

# ==============================================================================

