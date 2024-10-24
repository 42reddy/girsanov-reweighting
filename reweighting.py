from KineticsSandbox.potential.D1 import DoubleWell
from KineticsSandbox.potential.D1 import TripleWell
from KineticsSandbox.system import system
from KineticsSandbox.integrator import D1_stochastic
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from KineticsSandbox.potential import D1


l = 5   # length of the simulation box
n = 1   # number of particles
m = 1   # mass
T = 1  # temperature
kb = 1
N = int(1e7)  # num of iterations
x = 1.2         # initial positions
v = np.random.normal(0, np.sqrt(kb*T),n)    # initial velocities
dt = 0.01      # time step
h = 0.01
xi = 1    # friction coefficient


system = system.D1(m,x,v,T,xi,dt,h)       # initialize the system


param = [1,0,1]          # parameters for the double well potential
double_well = DoubleWell(param)      # initial class doublewell
triple_well = TripleWell([4])
potential = double_well.potential(np.linspace(-2,2,200))


class reweight():

    def __init__(self, n_states, tau, V_simulation, V_target):

        self.n_states = n_states
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target

    def generate(self):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = system.x
        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force

            D1_stochastic.EM(system, self.V_simulation, eta_k=eta[i])
            # update the system position
            X[i + 1] = system.x
        return X, eta

    def MSM(self, X, lag_time):

        """
        :param X: simulation path
        :param n_states: number of markov states
        :return: Transition matrix

        """

        bins = np.linspace(min(X) + 0.05 , max(X) - 0.05  , self.n_states - 1)
        discretized_path = np.digitize(X, bins)

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lag_time):
            count_matrix[discretized_path[i], discretized_path[i + lag_time]] += 1

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis= 1, keepdims=True)

        return transition_matrix, discretized_path

    def equilibrium_dist(self, transition_matrix):

        A = np.transpose(transition_matrix) - np.eye(self.n_states)

        A = np.vstack((A, np.ones(self.n_states)))
        b = np.zeros(self.n_states)
        b = np.append(b, 1)
        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        return pi

    def gradient(self, x):
        """

        :param x: position
        :param V_sim:  simulation potential
        :param V_target:  target potential
        :return: the gradient of bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)

    def reweighting_factor(self, X):

        """

        :param X: simulation path
        :param V_sim: simulation potential
        :param V_target: target potential
        :return: reweighting factors for each path
        """



        """ generate several paths using a sliding window method """

        paths = []
        for i in range(int((len(X) - self.tau))):
            paths.append(X[i : (i+1) + self.tau ])

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len(paths))
        for i in range(len(paths)):
            eta = np.zeros(3)
            delta_eta = np.zeros_like(eta)
            for j in range(3):

                eta[j] = ((paths[i][j + 1] - paths[i][j] - self.V_simulation.force_ana(paths[i][j])[0] * system.dt / system.xi_m) *(np.sqrt(m/(2*kb*T*dt))))

                #calculates delta eta from the gradient of bias potential
                delta_eta[j] = (double_well.force_ana(paths[i][j]) - triple_well.force_ana(paths[i][j])) * np.sqrt(dt/(2*kb*T*system.m))

            M[i] = np.exp((self.V_simulation.potential(paths[i][0]) - self.V_target.potential(paths[i][0]))/T) * (np.exp(-np.sum(eta*delta_eta)) * np.exp(-0.5 * np.sum(delta_eta**2)))

        return M, paths

    def reweighted_MSM(self, X, paths, M):

        """
        :param X: simulated path
        :param paths: generated paths from X
        :param M: rewighting factors
        :param lag_time: lag time for the markov model
        :return: reweighted transition matrix
        """

        tol = 0.2
        bins = np.linspace(min(X) + tol, max(X) - tol, self.n_states - 1)
        for i in range(len(paths)):
            paths[i] = np.digitize(paths[i], bins)

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(paths)):
            for j in range(len(paths[i]) - 1):
                count_matrix[paths[i][j], paths[i][j+1]] += 1 * M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix


