from scipy.fft import fft, ifft
from numpy import conj, arange, floor, real
from matplotlib.pyplot import plot, show

def data_filter(data, threshold):
    data_fft = fft(data)
    PSD = data_fft * conj(data_fft)/len(data)
    indices = PSD > threshold
    return real(ifft(data_fft * indices))

def show_data_peaks(data, dt):
    freq = (1/(dt*len(data))) * arange(len(data))
    data_fft = fft(data)
    PSD = data_fft * conj(data_fft)/len(data)
    L = arange(1, floor(len(data)/2), dtype='int')
    plot(freq[L], PSD[L])
    show()

def compare_filtered_data(x_space, data, threshold):
    plot(x_space, data, 'b')
    plot(x_space, real(data_filter(data, threshold)), 'orange')
    show()