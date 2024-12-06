from scipy import constants
from KineticsSandbox.integrator import D1_stochastic
import numpy as np

kb = 0.001 * constants.R

class reweight():

    def __init__(self, system, n_states, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target

    def generate(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        delta_eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = self.system.x
        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force
            delta_eta[i] = (np.sqrt(self.system.dt / (2 * kb * self.system.T * self.system.xi * self.system.m)) *
                            self.gradient(self.system.x)[0])
            D1_stochastic.EM(self.system, self.V_simulation, eta_k=eta[i])
            # update the system position
            X[i + 1] = self.system.x

        return X, eta, delta_eta

    def MSM(self, X, lag_time):

        """
        :param X: simulation path
        :param n_states: number of markov states
        :return: Transition matrix

        """
        lag_time = int(lag_time)
        bins = np.linspace(-2, 2, self.n_states - 1)
        discretized_path = np.digitize(X, bins)

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lag_time):
            count_matrix[discretized_path[i], discretized_path[i + lag_time]] += 1

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis= 1, keepdims=True)

        return transition_matrix

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

    def reweighting_factor(self, X, eta, delta_eta):

        """
        :param eta: random number at simulation potential
        :param delta_eta: random number at target potential
        :param X: simulation path
        :return: reweighting factors for each path
        """

        len_paths = int((len(X) - self.tau))  # length of the paths generated from sliding window method

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len_paths)
        for i in range(len_paths):
            """calculate eta and delta_eta for each path"""
            eta_ = eta[i : i + self.tau]
            delta_eta_ = delta_eta[i : i + self.tau]

            """calculate the reweighting factor"""

            M[i] = np.exp((self.V_simulation.potential(X[i]) - self.V_target.potential(
                    X[i])) / (kb * self.system.T)) * (np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2)))

        return M

    def reweighted_MSM(self, X, M, lag_time):

        """
        :param X: simulated path
        :param paths: generated paths from X
        :param M: rewighting factors
        :param lag_time: lag time for the markov model
        :return: reweighted transition matrix
        """

        bins = np.linspace(-2, 2, self.n_states - 1)
        len_paths = int(len(X) - self.tau)
        count_matrix = np.zeros((self.n_states, self.n_states))

        discretized_path = np.digitize(X, bins)

        for i in range(len_paths):
            path = discretized_path[i : i + self.tau+1]

            for j in range(int(self.tau - lag_time)):
                count_matrix[path[j], path[j + lag_time]] += 1 * M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix


