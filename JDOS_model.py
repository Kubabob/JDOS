from numba import jit
from scipy.optimize import differential_evolution
from numpy import array, exp, real, imag, sum, sqrt, log
from pandas import read_csv
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from time import perf_counter
from data_filtering import *
from scipy.signal import savgol_filter


@jit(nopython=True)
def fitness_count(sum_of_oscillators: array, e1_2nd_der_1: array, e2_2nd_der_1: array):
    '''
    RMSE of made oscillator and second derivatives
    :param sum_of_oscillators:
    :param e2_2nd_der_1:
    :param e1_2nd_der_1:
    :return:
    '''
    return sqrt(
        (sum((real(sum_of_oscillators) - e1_2nd_der_1) ** 2 + (imag(sum_of_oscillators) - e2_2nd_der_1) ** 2))/(2*len(e1_2nd_der_1)))

@jit(nopython=True)
def second_der(E, m, A, E_ck, gamma, ph_angle, discrete: bool):  # wzor z JAP2017
    '''
    2nd derivative from SCPM
    :param E:
    :param m:
    :param A:
    :param E_ck:
    :param gamma:
    :param ph_angle:
    :return:
    '''
    
    if discrete:
        if m < -(1/3):
            m = -0.5
        elif m > (1/3):
            m = 0.5
        else:
            m = 0


    if m != 0:
        return (-m * (m - 1) * A * exp(1j * ph_angle) * (E - E_ck + 1j * gamma)) ** (m - 2)
    else:
        return (A * exp(1j * ph_angle) * (E - E_ck + 1j * gamma)) ** (-2)

class JDOS:

    def __init__(self) -> None:
        self.AIC = {}
        self.BIC = {}
        self.dAIC = {}
        self.dBIC = {}

    def show_parameters(self, solution, path_to_save_score: str = None, max_value: float = 500):
        '''
        Shows parameters in a decent way
        :param max_value:
        :param solution:
        '''
        n = int(len(solution.x)/5)
        m = [solution.x[0*n + i] for i in range(n)]
        A = [solution.x[1*n + i] for i in range(n)]
        E_ck = [solution.x[2*n + i] for i in range(n)]
        gamma = [solution.x[3*n + i] for i in range(n)]
        phi = [solution.x[4*n + i] for i in range(n)]

        print(f'm: {m}')
        print(f'A: {A}')
        print(f'E_ck: {E_ck}')
        print(f'gamma: {gamma}')
        print(f'phi: {phi}')

        if path_to_save_score:
            if solution.fun < max_value:
                with open(f'{path_to_save_score}.txt', mode='a') as f:
                    f.write(f'\n{n}\n{m}\n{A}\n{E_ck}\n{gamma}\n{phi}\n{solution.fun}\n')



    def merge_oscillators(self, oscillators: list[array]):
        '''
        Merges SCPM oscillators into one
        :param oscillators:
        :return np.array:
        '''
        m = []
        A = []
        E_ck = []
        gamma = []
        phi = []
        for oscillator in oscillators:
            n = int(len(oscillator)/5)
            m.extend([oscillator[0 * n + i] for i in range(n)])
            A.extend([oscillator[1 * n + i] for i in range(n)])
            E_ck.extend([oscillator[2 * n + i] for i in range(n)])
            gamma.extend([oscillator[3 * n + i] for i in range(n)])
            phi.extend([oscillator[4 * n + i] for i in range(n)])
        else:
            ret = []
            ret.extend(m)
            ret.extend(A)
            ret.extend(E_ck)
            ret.extend(gamma)
            ret.extend(phi)
            return array(ret)


    def fitness_func(self, solution, n, eV_space, e1_2nd_der_1, e2_2nd_der_1, discrete: bool):
        '''
        Fitness function

        sadly can't be optimized with JIT
        at least I couldn't do it now
        :param solution:
        :param n:
        :param eV_space:
        :param e1_2nd_der_1:
        :param e1_2nd_der_2:
        :param e2_2nd_der_1:
        :param e2_2nd_der_2:
        :return:
        '''
        suma = array(
            [[second_der(E, solution[0 * n + i], solution[1 * n + i], solution[2 * n + i], solution[3 * n + i],
                        solution[4 * n + i], discrete) for E in eV_space] for i in range(n)]).sum(axis=0)

        return fitness_count(suma, e1_2nd_der_1, e2_2nd_der_1)

    def fitting_method(self,
                       filepath: str, 
                        left: int,
                        right: int,
                        model_order: int,
                        filter_data: bool,
                        plot: bool = False,
                        discrete_m: bool = False,
                        separator: str = ';',
                        threads: int = 1,
                        population_size: int = 50,
                        single_sol: bool = True,
                        window_length: int = 10,
                        poly_order: int = 0,
                        results_path: str = 'wyniki'):

        if single_sol:
            updating = 'immediate'
        else:
            updating = 'deferred'

        input_data = read_csv(filepath, sep=separator)

        e1 = array(input_data['e1'][left:right:-1])
        e2 = array(input_data['e2'][left:right:-1])
        eV = array(input_data['eV'][left:right:-1])

        if filter_data:
            eV = savgol_filter(eV, window_length, poly_order)
            e1 = savgol_filter(e1, window_length, poly_order)
            e2 = savgol_filter(e2, window_length, poly_order)
            e1_second_der = savgol_filter(((UnivariateSpline(eV, e1, s=0)).derivative(2)(eV)), window_length, poly_order)
            e2_second_der = savgol_filter(((UnivariateSpline(eV, e2, s=0)).derivative(2)(eV)), window_length, poly_order)
        else:
            e1_second_der = (UnivariateSpline(eV, e1, s=0)).derivative(2)(eV)
            e2_second_der = (UnivariateSpline(eV, e2, s=0)).derivative(2)(eV)



        bounds = []
        bounds.extend([(-0.5, 0.5) for _ in range(model_order)])  # m
        bounds.extend([(0.1, 15) for _ in range(model_order)])  # A
        if eV[0] - 2 < 0:
            bounds.extend([(0.1, eV[-1]+2) for _ in range(model_order)])  # E_ck
        else:            
            bounds.extend([(eV[0] - 2, eV[-1]+2) for _ in range(model_order)])  # E_ck
        bounds.extend([(0.01, 10) for _ in range(model_order)])  # gamma
        bounds.extend([(-3.14, 3.14) for _ in range(model_order)])  # phase angle


        st = perf_counter()
        result = differential_evolution(self.fitness_func,
                                        bounds,
                                        args=[model_order, eV, e1_second_der, e2_second_der, discrete_m],
                                        workers=threads,
                                        popsize=population_size,
                                        updating=updating)
        
        print(f'It took: {perf_counter()-st}')
        print(f'Result: {list(result.x)}')
        self.show_parameters(result, results_path, 1)
        print(f'Fitness value(max 0): {result.fun}\n')

        if model_order in self.AIC.keys():
            self.AIC[model_order].append(len(eV) * log(result.fun / len(eV)) + 2 * len(result.x))
        else:
            self.AIC[model_order] = [len(eV)*log(result.fun/len(eV)) + 2*len(result.x)]

        if model_order in self.BIC.keys():
            self.BIC[model_order].append(len(eV) * log(result.fun / len(eV)) + len(result.x) * log(model_order))
        else:
            self.BIC[model_order] = [len(eV)*log(result.fun/len(eV)) + len(result.x)*log(len(eV))]


        if plot:
            oscillators_sum = array([[second_der(E, result.x[0 * model_order + i], result.x[1 * model_order + i], result.x[2 * model_order + i],
                                                    result.x[3 * model_order + i], result.x[4 * model_order + i], discrete_m) for E in eV] for i in
                                        range(model_order)]).sum(axis=0)


            plt.plot(eV, real(oscillators_sum), 'green')
            plt.plot(eV, e1_second_der, 'g--')

            plt.plot(eV, imag(oscillators_sum), 'r')
            plt.plot(eV, e2_second_der, 'r--')

            plt.gca().invert_xaxis()
            plt.yscale(value='linear')
            plt.grid()
            plt.show()

    def plot(self,
             result,
             discrete_m: bool,
             eV,
             e1_second_der,
             e2_second_der):
        model_order = int(len(result)/5)
        oscillators_sum = array([[second_der(E, result[0 * model_order + i], result[1 * model_order + i], result[2 * model_order + i],
                                                    result[3 * model_order + i], result[4 * model_order + i], discrete_m) for E in eV] for i in
                                        range(model_order)]).sum(axis=0)


        plt.plot(eV, real(oscillators_sum), 'green')
        plt.plot(eV, e1_second_der, 'g--')

        plt.plot(eV, imag(oscillators_sum), 'r')
        plt.plot(eV, e2_second_der, 'r--')

        plt.gca().invert_xaxis()
        plt.yscale(value='linear')
        plt.grid()
        plt.show()

    def compare_model_order(self,
                        order_bounds: list[int],
                        tests_per_order: int,
                        filepath: str, 
                        left: int,
                        right: int,
                        filter_data: bool,
                        plot: bool = False,
                        discrete_m: bool = False,
                        separator: str = ';',
                        threads: int = 1,
                        population_size: int = 30,
                        window_length: int = 8,
                        poly_order: int = 2):
        
        
        
        for order in range(order_bounds[0], order_bounds[1]):
            for _ in range(tests_per_order):
                self.fitting_method(filepath,
                            left,
                            right,
                            order,
                            filter_data,
                            plot,
                            discrete_m,
                            separator,
                            threads,
                            population_size,
                            False,
                            window_length,
                            poly_order)
                
        else:
            for IC in self.AIC.keys():
                self.dAIC[IC] = abs((min(self.AIC.get(IC)) - min(self.AIC.values()))[0])
                self.dBIC[IC] = abs((min(self.BIC.get(IC)) - min(self.BIC.values()))[0])
            else:
                print(f'dAIC: {self.dAIC}\n')
                print(f'dBIC: {self.dBIC}\n')
                sorted_dAIC_keys = sorted(self.dAIC.keys())
                sorted_dBIC_keys = sorted(self.dBIC.keys())
                plt.plot(sorted_dAIC_keys, [self.dAIC[x] for x in sorted_dAIC_keys], 'b--', marker='o')
                plt.plot(sorted_dBIC_keys, [self.dBIC[x] for x in sorted_dBIC_keys], 'r--', marker='o')
                plt.show()
