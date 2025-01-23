from scipy import constants
from KineticsSandbox.integrator import D1_stochastic
import numpy as np
from KineticsSandbox.potential import D1
import tensorflow as tf

kb = 0.001 * constants.R

class reweight():

    def __init__(self, system, n_states, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target
        self.model = tf.keras.models.load_model('force_prediction_model.h5')

    def gradient(self, x):
        """

        :param x: position
        :return: the gradient of bias potential, bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)


    def generate1(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        delta_eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = self.system.x

        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force
            force = self.model.predict(X[i].reshape(-1,1))
            delta_eta[i] = (np.sqrt(self.system.dt / (2 * kb * self.system.T * self.system.xi * self.system.m)) *
                            -force)
            #D1_stochastic.EM(self.system, self.V_simulation, eta_k=eta[i])
            force = self.V_simulation.force_ana(X[i]) + force
            # update the system position
            X[i + 1] = X[i] + force*self.system.dt / self.system.xi_m + self.system.sigma * np.sqrt(self.system.dt) * eta[i]

        return X, eta, delta_eta

    def metadynamics_generate(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        delta_eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = self.system.x
        times_added =[]

        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force

            #D1_stochastic.EM(self.system, self.V_simulation, eta_k=eta[i])
            force = self.V_simulation.force_ana(X[i])
            # update the system position
            X[i + 1] = X[i] + force*self.system.dt / self.system.xi_m + self.system.sigma * np.sqrt(self.system.dt) * eta[i]

            if i%100 ==0:
                times_added.append(X[i+1])

        return X, eta, delta_eta, times_added


    def metadynamics(self, N):

        eta = np.zeros(N)  # Random forces
        delta_eta = np.zeros(N)  # Random number difference
        X = np.zeros(N)  # Position array
        X[0] = self.system.x  # Initial position
        times_added = []  # To store positions where bias is added
        bias_potential = np.zeros(N)

        for i in range(N - 1):
            # Random force for stochastic dynamics
            eta[i] = np.random.normal(0, 1)

            double_well_force = -((2 * (X[i]**3 - (3/2) * X[i])) * (3 * X[i]**2 - (3/2)) - 3 * X[i]**2 + 1)

            # Update bias force and bias potential based on the new position X[i]
            bias_force = 0
            if len(times_added) > 0:
                x = X[i] - np.array(times_added)
                bias_force = 0.0001 * np.sum((x/0.005) * np.exp(-x ** 2 / 0.01))

            # Store the bias potential at the current position
            #if len(times_added) > 0:
                #bias_potential[i] = 0.001 * np.sum(np.exp(- (X[i] - np.array(times_added)) ** 2 / 8))

            # Calculate delta_eta using the bias force
            delta_eta[i] = (np.sqrt(self.system.dt / (2 * kb * self.system.T * self.system.xi * self.system.m)) *
                            -bias_force)

            # Total force is the sum of the double-well force and bias force
            force = double_well_force + bias_force

            # Overdamped Langevin dynamics with Euler-Maruyama scheme
            X[i + 1] = (X[i] +
                        force / self.system.xi_m * self.system.dt +
                        self.system.sigma * np.sqrt(self.system.dt) * eta[i])

            # Add bias every 50 timesteps
            if i % 100 == 0:
                    times_added.append(X[i])

        return X, eta, delta_eta, times_added

    def MSM(self, X, lagtime):

        """
        :param X: simulation path
        :return: Transition matrix

        """

        bins = np.linspace(min(X), max(X), self.n_states)
        discretized_path = np.digitize(X, bins, right=False) - 1

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lagtime):
            count_matrix[discretized_path[i], discretized_path[i + lagtime]] += 1

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

    def reweighting_factor(self, X, eta, delta_eta, lagtime, times_added):

        """
        :param eta: random number at simulation potential
        :param delta_eta: random number at target potential
        :param X: simulation path
        :return: reweighting factors for each path
        """

        len_paths = int((len(X) - lagtime))  # length of the paths generated from sliding window method

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len_paths)
        for i in range(len_paths):
            """calculate eta and delta_eta for each path"""
            eta_ = eta[i : i + lagtime]
            delta_eta_ = delta_eta[i : i + lagtime]


            """calculate the reweighting factor"""

            M[i] = np.exp(-0.001 * np.sum(np.exp(- (X[i] - np.array(times_added)) ** 2 / 0.01)) / (kb * self.system.T)) * (np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2)))

        return M

    def reweighted_MSM(self, X, M, lagtime):

        """
        :param X: simulated path
        :param M: rewighting factors
        :param lagtime: lag time for the markov model
        :return: reweighted transition matrix
        """

        # X = X[(X <= 1.6) & (X >= -1.7)]
        bins = np.linspace(min(X), max(X), self.n_states + 1, endpoint=True)
        discretized_path = np.zeros(len(X))

        for i in range(self.n_states):
            if i == 0:
                discretized_path[(X >= bins[0]) & (X < bins[1])] = 0
            elif i == self.n_states - 1:
                discretized_path[(X >= bins[self.n_states - 1]) & (X <= bins[self.n_states])] = self.n_states - 1
            else:
                discretized_path[(X >= bins[i]) & (X < bins[i+1])] = i
        discretized_path = discretized_path.astype(int)



        count_matrix = np.zeros((self.n_states, self.n_states))
        #discretized_path[discretized_path == self.n_states+1] = self.n_states
        #discretized_path[discretized_path == 0] = 1
        #discretized_path -=

        len_paths = int(len(X) - lagtime)
        for i in range(len_paths):
            path = discretized_path[i: i + lagtime]

            '''transitions = path[:self.tau - lag_time], path[lag_time:]  # Two slices for transitions

            # Count the transitions
            for start, end in zip(*transitions):
                count_matrix[start, end] += M[i]'''

            count_matrix[path[0], path[lagtime - 1]] += M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix

    def implied_timescales(self, X, X2, eta, delta_eta, lagtimes):

        lagtimes = lagtimes.astype(int)

        eigen3 = np.zeros(len(lagtimes))
        eigen2 = np.zeros(len(lagtimes))
        eigen1 = np.zeros(len(lagtimes))

        for i in range(len(lagtimes)):

            T_ = self.MSM(X, lagtimes[i])
            T_ = np.nan_to_num(T_, nan=0)
            eig = np.linalg.eigvals(T_)
            eig = np.sort(eig)
            print(eig[-2])
            eigen1[i] = -lagtimes[i] / np.log(eig[-2])

            T_ = self.MSM(X2, lagtimes[i])
            T_ = np.nan_to_num(T_, nan=0)
            eig = np.linalg.eigvals(T_)
            eig = np.sort(eig)
            print(eig[-2])
            eigen2[i] = -lagtimes[i] / np.log(eig[-2])

            M = self.reweighting_factor(X, eta, delta_eta, lagtimes[i])
            T_reweighted = self.reweighted_MSM(X, M, lagtimes[i])
            T_reweighted = np.nan_to_num(T_reweighted, nan=0)
            eig = np.linalg.eigvals(T_reweighted)
            eig = np.sort(eig)
            print(eig[-2])
            eigen3[i] = -lagtimes[i] / np.log(eig[-2])

        return eigen1, eigen2, eigen3


