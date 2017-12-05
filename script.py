
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift

import scipy

from skimage.util.shape import view_as_windows

from scipy.signal import bartlett, hanning, hamming, blackman, freqz

import matplotlib.pyplot as plt 

# ==============================================================================
# Plotting/Utility Functions
# ==============================================================================

"""
Simple function to convert a value, array, or matrix to dB.

x :: Input sequence

Note: Multiplies by 10 since it assumes power values will already be squared
"""
def dB(x):
    return 10*np.log10(x)

# ------------------------------------------------------------------------------

"""
Plots power spectrum plot

dat :: Input matrix
title :: Title to display
line :: y-coordinate to plot noise floor line
df :: Hz per column

Note: See report for default values
"""
def powplot(dat,title='Power Plot',line=None,df=500):
    plt.plot(dB(dat))
    if line != None:
        plt.plot(np.array(dat.shape[0]*[dB(line)]))

    plt.title(title)

    plt.xlabel('Freq [%dHz/col]'%df)
    plt.ylabel('Power (dB)')

    plt.grid()
    axes = plt.gca()
    axes.set_xlim([0,dat.shape[0]])

    plt.show()

# ------------------------------------------------------------------------------

"""
Plots provided data as heatplot

dat :: Input matrix
title :: Title to display
df :: Hz per column
dt :: ms per row
pr :: extra points to plot, row-coord
pc :: extra points to plot, col-coord

Note: Produce band occupancy detector if provided a matrix of boolean values
Note: See report for default values
"""
def heatplot(dat,title='HeatMap',df=500,dt=1,pr=None,pc=None):
    plt.imshow(dB(dat), cmap='hot', interpolation='nearest')

    if pr != None and pc != None:
        plt.plot(pc,pr,'o',markersize=10)

    plt.title(title)
    plt.xlabel('Freq [%dHz/col]'%df)
    plt.ylabel('Time [%dms/row]'%dt)

    plt.grid()
    axes = plt.gca()
    axes.set_xlim([0,dat.shape[1]])
    axes.set_ylim([dat.shape[0],0])

    plt.show()

# ==============================================================================



# ==============================================================================
# Project 2 Section 1 Part a Functions
# ==============================================================================

"""
Time aliases an input sequence to desired length.

dat :: Input data sequence to be aliased
n :: Desired length of aliased sequence

Assertion: The length of data sequence must be greater than the desired
length of the aliased output sequence

Used by Project 2 Section 1 Part a
"""
def time_alias(dat,n):
    # ensures length of data sequence is greater than desired output length
    assert len(dat) > n

    # handles complete alias wrap around
    aliased = sum(view_as_windows(dat,n,n))

    # handles incomplete alias wrap around
    overlen = len(dat) % n
    if overlen != 0:
        extra = np.zeros(n,dtype=aliased.dtype)
        extra[:overlen] = dat[-overlen:] 
        aliased += extra

    return aliased

# ------------------------------------------------------------------------------

"""
Performs a TD-DTFT and returns a matrix of DFT samples at each time window

dat :: Input time domain data sequence
winlen :: Desired window length
winoverlap :: Desired step size between windows
dftsize :: Desired number of samples of DTFT to be computed
winty :: Desired windowing sequence as a string

-= winty options =-
>> 'rect'
>> 'bartlett'
>> 'hann'
>> 'hamming'
>> 'blackman'

Note: See report for justifications of default values

Used by Project 2 Section 1 Part a
"""
def td_dft(dat, winlen=4096, winoverlap=2048, dftsize=4096, winty='blackman'):
    # calculate the views
    views = view_as_windows(dat,winlen,winlen-winoverlap)

    # generate desired window
    if winty == 'rect':
        win = np.ones(winlen)
    elif winty == 'bartlett':
        win = bartlett(winlen)
    elif winty == 'hann':
        win = hanning(winlen)
    elif winty == 'hamming':
        win = hamming(winlen)
    elif winty == 'blackman':
        win = blackman(winlen)
    else:
        assert False # invalid winty


    # apply window sequence to views
    views = [ v*win for v in views ]

    # computes time aliasing to input sequences if needed
    if winlen > dftsize:
        views = [ time_alias(v,dftsize) for v in views ]

    # apply fft and fftshift to all views
    dfts = [ fftshift(fft(v,dftsize)) for v in views ]

    return np.array(dfts)

# ------------------------------------------------------------------------------

"""
Creates the matrix of spectrogram data by using td_dft

dat :: Input time domain data sequence
winlen :: Desired window length
winoverlap :: Desired step size between windows
dftsize :: Desired number of samples of DTFT to be computed
winty :: Desired windowing sequence as a string

-= winty options =-
>> 'rect'
>> 'bartlett'
>> 'hann'
>> 'hamming'
>> 'blackman'

Note: See report for justifications of default values

Used by Project 2 Section 1 Part a
"""

def spectrogram(dat, winlen=4096, winoverlap=2048, dftsize=4096, winty='blackman'):
    # calculate TD-DFT
    mat = td_dft(dat, winlen, winoverlap, dftsize, winty=winty)

    # calculate |.|^2
    mag = np.abs(mat)
    sqr = mag*mag

    return sqr

# ==============================================================================



# ==============================================================================
# Project 2 Section 1 Part b Functions
# ==============================================================================

"""
Calculates a Power Spectrum Estimate from a spectrogram by averaging
periodograms (i.e. rows)

spec :: Spectrogram input

Used by Project 2 Section 1 Part b
"""
def power_est_periodogram(spec):
    # averages periodogram (i.e. spectrogram rows)
    pgram = np.sum(spec,axis=0) / spec.shape[0]
    return pgram

# ------------------------------------------------------------------------------

"""
Calculates a Power Spectrum Estimate by creating an All-Pole model by
calculating the autocorrelation function of the input sequence and using it
to solve the Yule-Walker equations.

dat :: Time series input sequence
order :: Desired order of All-Pole Model

Note: An undiagnosed bug exists in this function - it does not work as intended

Alternative approach for Project 2 Section 1 Part b
"""
def power_est_allpole(dat,order):
    # calculate needed values of autocorrelation function
    rss = [ np.correlate(dat,np.conj(dat[i:]))[0] for i in range(order+1) ]

    # create the autocorrelation matrix used by Yule-Walker equations
    automat = [ [ rss[abs(i-j)] for j in range(order) ] for i in range(order) ]

    # invert autocorrelation matrix to solve for alpha parameters
    mat = np.linalg.inv(np.array(automat))

    # autocorrelation vector of the Yule-Walker equations
    vec = np.array(rss[1:])

    # solving for the alpha parameters
    alpha = np.matmul(mat,vec)

    # generate frequency response of All-Pole filter
    w,h = freqz([1],[1]+list(-1*alpha),whole=True)

    # poor man's fftshift
    h = np.array(list(h[len(h)//2:]) + list(h[:len(h)//2]))

    # plot
    plt.plot(w/pi-1,2*dB(h))

    plt.title('All-Pole Model (Order %d)' % order)
    plt.xlabel('Normalized Freq')
    plt.ylabel('Power (dB)')

    plt.grid()
    axes = plt.gca()
    axes.set_xlim([-1, 1])

    plt.show()

# ==============================================================================



# ==============================================================================
# Project 2 Section 2 Part a Functions
# ==============================================================================

"""
Estimates the noise floor by averaging smallest 20% (by default) of spectrogram
cells together.

spec :: Spectrogram input matrix
percent :: Percentage of minimum values to average together

Assertion: Parameter percent is in range (0,1]

Used by Project 2 Section 2 Part a
"""
def noise_est(spec, percent=.2):
    assert percent > 0 and percent <= 1
    sort = np.sort(spec.flatten())
    subset = sort[:int(len(sort)*percent)]
    avg = np.mean(subset)
    return avg

# ==============================================================================



# ==============================================================================
# Project 2 Section 3 Part a Functions
# ==============================================================================

"""
Returns boolean matrix of whether each spectrogram cell exceeds the sigma*eta
threshold value

spec :: Spectrogram input matrix
sigma :: Sigma value component of threshold
eta :: Eta value component of threshold

Used by Project 2 Section 3 Part a
"""
def band_occupancy(spec, sigma, eta):
    # create map function
    thresh = np.vectorize(lambda x: x >= sigma*eta)
    
    # apply map function
    detects = thresh(spec)

    return detects

# ==============================================================================



# ==============================================================================
# Project 2 Section 3 Part b Functions
# ==============================================================================

"""
Finds the n greatest points in spectrogram that are at least dr rows apart
and dc columns apart. Confusion implementation is so it runs fast.

spec :: Spectrogram input matrix
n :: Number of points searching for
dr :: Minimum row distance between accepted points
dc :: Minimum column distance between accepted points

Assumption: At least n points that dr and dc distance apart exist

Used by Project 2 Section 3 Part b
"""
def strongest(spec, n=5, dr=2, dc=8):
    L = spec
    accepted = []

    while len(accepted) < 5:
        sort = np.sort(L.flatten())
        big = sort[-1]

        r,c = np.where(L == big)
        r = r[0]
        c = c[0]
        
        rl = r-dr
        if rl < 0:
            rl = 0
        ru = r+dr+1
        if ru > L.shape[0]:
            ru = L.shape[0]

        cl = c-dc
        if cl < 0:
            cl = 0
        cu = c+dc+1
        if cu > L.shape[1]:
            cu = L.shape[1]

        L = np.delete(L, np.s_[rl:ru],0)
        L = np.delete(L, np.s_[cl:cu],1)

        accepted.append(big)

    data = []
    for accept in accepted:
        r,c = np.where(spec == accept)
        assert len(r) == 1
        data.append( (accept,r[0],c[0]) )

    return data

# ==============================================================================
# Processing Function
# ==============================================================================

def process(samples,percent=.2,eta=1000):
    # winlen, winoverlap, dftsize, winty
    spec = spectrogram(samples)
    heatplot(spec,title='Spectrogram')

    pest = power_est_periodogram(spec)
    powplot(pest,title='Power Spectrum Estimate - Periodogram Avg')

    sigma = noise_est(spec,percent)
    print('sigma:', sigma)
    powplot(pest,title='Sigma Noise Floor Estimate',line=sigma)

    detects = band_occupancy(spec, sigma, eta)
    heatplot(detects,title='Band Occupancy')

    # commented out to use precomputed values (to save runtime)
    strong = strongest(spec)
    print('strongest (p,r,c):',strong)

    # precomputed values for strong (to save runtime)
    #strong = [
    #    (4501908.5024756445, 3136, 2523),
    #    (1038229.7014512023, 3141, 2416),
    #    (434085.43847087101, 8040, 622),
    #    (84952.945846237766, 8044, 947),
    #    (71708.022026160368, 8047, 2532)
    #]
    
    pr = [ r for _,r,_ in strong ]
    pc = [ c for _,_,c in strong ]
    heatplot(spec,title='5 Strongest Signals',pr=pr,pc=pc)

# ==============================================================================


# ==============================================================================
# Data Set Functions
# ==============================================================================

def ld_file(infile):
    return scipy.fromfile(open(infile), dtype=scipy.complex64)

# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    # data, percent noise estimate, eta
    #process(ld_file('blind_test.raw'))
    process(ld_file('blind_test_project02.raw'))

    #power_est_allpole(ld_file('blind_test_project02.raw'),10)

# ==============================================================================

