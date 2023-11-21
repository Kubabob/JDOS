from numba import jit
from scipy.optimize import differential_evolution
from numpy import array, exp, real, imag, sum, sqrt, log
from pandas import read_csv
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from time import perf_counter
from data_filtering import *


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
        sum((real(sum_of_oscillators) - e1_2nd_der_1) ** 2 + (imag(sum_of_oscillators) - e2_2nd_der_1) ** 2))

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
                        last_index: int,
                        model_order: int,
                        filter_data: bool,
                        threshold_e1: int = 0,
                        threshold_e2: int = 0,
                        plot: bool = False,
                        discrete_m: bool = False,
                        separator: str = ';',
                        threads: int = 1,
                        population_size: int = 30,
                        single_sol: bool = True):

        if single_sol:
            updating = 'immediate'
        else:
            updating = 'deferred'

        input_data = read_csv(filepath, sep=separator)

        e1 = array(input_data['e1'][last_index::-1])
        e2 = array(input_data['e2'][last_index::-1])
        eV = array(input_data['eV'][last_index::-1])

        if filter_data:
            e1_second_der = data_filter((UnivariateSpline(eV, e1, s=0)).derivative(2)(eV), threshold_e1)
            e2_second_der = data_filter((UnivariateSpline(eV, e2, s=0)).derivative(2)(eV), threshold_e2)
        else:
            e1_second_der = (UnivariateSpline(eV, e1, s=0)).derivative(2)(eV)
            e2_second_der = (UnivariateSpline(eV, e2, s=0)).derivative(2)(eV)

        bounds = []
        bounds.extend([(-0.5, 0.5) for _ in range(model_order)])  # m
        bounds.extend([(0.1, 30) for _ in range(model_order)])  # A
        if eV[0] - 2 < 0:
            bounds.extend([(0.1, eV[-1]+2) for _ in range(model_order)])  # E_ck
        else:            
            bounds.extend([(eV[0] - 2, eV[-1]+2) for _ in range(model_order)])  # E_ck
        bounds.extend([(0.1, 10) for _ in range(model_order)])  # gamma
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
        self.show_parameters(result, 'wyniki', 1000)
        print(f'Fitness value(max 0): {result.fun}\n')

        if model_order in self.AIC.keys():
            self.AIC[model_order].append(model_order * log(result.fun / model_order) + 2 * len(result.x))
        else:
            self.AIC[model_order] = [model_order*log(result.fun/model_order) + 2*len(result.x)]

        if model_order in self.BIC.keys():
            self.BIC[model_order].append(model_order * log(result.fun / model_order) + len(result.x) * log(model_order))
        else:
            self.BIC[model_order] = [model_order*log(result.fun/model_order) + len(result.x)*log(model_order)]


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
            plt.ylim(bottom=-120, top=120)
            plt.grid()
            plt.show()


    def compare_model_order(self,
                        order_bounds: list[int],
                        tests_per_order: int,
                        filepath: str, 
                        last_index: int,
                        filter_data: bool,
                        threshold_e1: int = 0,
                        threshold_e2: int = 0,
                        plot: bool = False,
                        discrete_m: bool = False,
                        separator: str = ';',
                        threads: int = 1,
                        population_size: int = 30):
        
        
        
        for order in range(order_bounds[0], order_bounds[1]):
            for _ in range(tests_per_order):
                self.fitting_method(filepath,
                            last_index,
                            order,
                            filter_data,
                            threshold_e1,
                            threshold_e2,
                            plot,
                            discrete_m,
                            separator,
                            threads,
                            population_size,
                            False)
                
        else:
            for IC in self.AIC.keys():
                self.dAIC[IC] = abs((min(self.AIC.get(IC)) - min(self.AIC.values()))[0])
                self.dBIC[IC] = abs((min(self.BIC.get(IC)) - min(self.BIC.values()))[0])
            else:
                print(f'dAIC: {self.dAIC}\n')
                print(f'dBIC: {self.dBIC}\n')
                plt.plot(list(self.dAIC.keys()), list(self.dAIC.values()), 'b--', list(self.dAIC.keys()), list(self.dAIC.values()), 'bo')
                #plt.scatter(list(self.dAIC.keys()), list(self.dAIC.values()), 'b')
                plt.plot(list(self.dBIC.keys()), list(self.dBIC.values()), 'r--', list(self.dBIC.keys()), list(self.dBIC.values()), 'ro')
                #plt.scatter(list(self.dBIC.keys()), list(self.dBIC.values()), 'r')
                plt.show()
