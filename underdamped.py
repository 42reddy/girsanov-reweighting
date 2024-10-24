from KineticsSandbox.system import system
from KineticsSandbox.integrator import D1_stochastic
import numpy as np


class underdamped():

    def __init__(self, system, n_states, lagtime, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states
        self.lagtime = lagtime
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target

    def gradient(self, x):
        """

        :param x: position
        :return: the gradient of bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)

    def BAOAB(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the BAOAB algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.O_step(self.system, eta_k=eta_k)
        delta_eta = np.exp(-self.system.xi * self.system.dt) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (
                        np.sqrt(self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt))))
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.B_step(self.system, potential, half_step=True)

        return delta_eta

    def BOAOB(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the BOAOB algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta = np.exp(-self.system.xi * self.system.dt / 2) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt / 2))))
        D1_stochastic.A_step(self.system)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta1 = (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)

        return delta_eta, delta_eta1

    def OABAO(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the OABAO algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta = (self.system.dt / 2) * (self.gradient(self.system.x))[0] / (
            np.sqrt(self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt / 2))))
        D1_stochastic.A_step(self.system)
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta1 = np.exp(-self.system.xi * self.system.dt / 2) * (self.system.dt / 2) *(self.gradient(self.system.x))[
                        0] / (np.sqrt(self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt / 2))))

        return delta_eta, delta_eta1

    def generate(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        delta_eta = np.zeros(N)
        #delta_eta1 = np.zeros(N)
        eta1 = np.zeros(N)
        #eta2 = np.zeros(N)
        X = np.zeros(N)
        v = np.zeros(N)
        X[0] = self.system.x
        for i in range(N - 1):
            eta1[i] = np.random.normal(0, 1)  # random number corresponding to random force
            #eta2[i] = np.random.normal(0,1)
            delta_eta[i] = self.BAOAB(self.V_simulation, eta_k=eta1[i])
            # update the system position
            X[i + 1] = self.system.x
            v[i + 1] = self.system.v

        return X, v, eta1,  delta_eta

    def MSM(self, X, lag_time):

        """
        :param X: simulation path
        :return: Transition matrix

        """

        bins = np.linspace(min(X) + 0.05, max(X) - 0.05, self.n_states - 1)
        discretized_path = np.digitize(X, bins)

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lag_time):
            count_matrix[discretized_path[i], discretized_path[i + lag_time]] += 1

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix, discretized_path

    def equilibrium_dist(self, transition_matrix):
        A = np.transpose(transition_matrix) - np.eye(self.n_states)

        A = np.vstack((A, np.ones(self.n_states)))
        b = np.zeros(self.n_states)
        b = np.append(b, 1)
        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        return pi

    def reweighting_factor(self, X, eta, delta_eta, eta1=None, delta_eta1=None):

        """
        :param delta_eta1: second random number at target potential
        :param eta1: second random number at simulation
        :param eta: random number at simulation potential
        :param delta_eta: random number at target potential
        :param X: simulation path
        :return: reweighting factors for each path
        """

        len_paths = int((len(X) - self.tau) / self.lagtime)  # length of the paths generated from sliding window method

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len_paths)
        for i in range(len_paths):
            """calculate eta and delta_eta for each path"""
            eta_ = eta[i*self.lagtime : (i + 1) * self.lagtime + self.tau: self.lagtime]
            delta_eta_ = delta_eta[i*self.lagtime : (i + 1) * self.lagtime + self.tau: self.lagtime]

            """calculate the reweighting factor"""

            if delta_eta1 is not None:    # switch between schemes with single and two random numbers
                eta_1 = eta1[i * self.lagtime: (i + 1) * self.lagtime + self.tau: self.lagtime]
                delta_eta_1 = delta_eta1[i * self.lagtime: (i + 1) * self.lagtime + self.tau: self.lagtime]

                M[i] = np.exp((self.V_simulation.potential(X[i*self.lagtime]) - self.V_target.potential(X[i*self.lagtime])) / self.system.T) * (
                           np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2))) * (
                           np.exp(-np.sum(eta_1 * delta_eta_1)) * np.exp(-0.5 * np.sum(delta_eta_1 ** 2)))
            else:
                M[i] = np.exp((self.V_simulation.potential(X[i * self.lagtime]) - self.V_target.potential(
                    X[i * self.lagtime])) / self.system.T) * (np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2)))

        return M

    def reweighted_MSM(self, X, M):

        """
        :param X: simulated path
        :param paths: generated paths from X
        :param M: rewighting factors
        :param lag_time: lag time for the markov model
        :return: reweighted transition matrix
        """

        tol = 0
        bins = np.linspace(min(X) + tol, max(X) - tol, self.n_states - 1)
        len_paths = int((len(X) - self.tau) / self.lagtime)
        count_matrix = np.zeros((self.n_states, self.n_states))
        path_len = int(self.tau/self.lagtime)

        for i in range(len_paths):
            path = np.digitize(X[i * self.lagtime : (i + 1) * self.lagtime + self.tau : self.lagtime], bins)

            for j in range(path_len):
                count_matrix[path[j], path[j + 1]] += 1 * M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix
