from pandas import read_csv
from numpy import array, mean
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

SiDF = read_csv('./Si.csv', sep=r'\t')

e1 = savgol_filter(array(SiDF['e1'][::-1]), 8, 2)
e2 = savgol_filter(array(SiDF['e2'][::-1]), 8, 2)
eV = array(SiDF['eV'][::-1])

e1_second_der = (UnivariateSpline(eV, e1, s=0)).derivative(2)(eV)
e2_second_der = (UnivariateSpline(eV, e2, s=0)).derivative(2)(eV)


def loss_function(exp_e1, exp_e2, mod_e1, mod_e2, points_number):
    return (((exp_e1-mod_e1)**2 + (exp_e2-mod_e2)**2)/(2*points_number)).mean(axis=1)