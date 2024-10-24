from KineticsSandbox.potential.D1 import DoubleWell
from KineticsSandbox.potential.D1 import TripleWell
from KineticsSandbox.system import system
from KineticsSandbox.integrator import D1_stochastic
import numpy as np
import matplotlib.pyplot as plt
from reweighting import reweight
import scipy.constants as const

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


instance = reweight(40,8, triple_well,double_well)   # intiate the class reweighting

#generates simulation path

X, eta = instance.generate()
plt.hist(X, bins=1000)

# generate the transition matrix
T_M, path = instance.MSM(X,10 )

# calculates and plots the first eigen vector
eig_vals, eig_vecs = np.linalg.eig(T_M.T)
left_eig = eig_vecs.T
plt.plot(left_eig[1])

# plots the equilibrium distribution
pi1 = instance.equilibrium_dist(T_M)
plt.title('equilibrium distribution')
plt.plot(pi1)


# calculates the reweighting factors for each path
M, paths = instance.reweighting_factor(X)
plt.xlabel('path index')
plt.ylabel('reweighting factor M')
plt.plot(M)

# calulates the reweighted transition matrix
reweighted_matrix = instance.reweighted_MSM(X= X, paths= paths, M= M)
pi = instance.equilibrium_dist(reweighted_matrix)
pi = pi/np.sum(pi)
plt.title('reweighted equilibrium distribution')
plt.plot(pi)





# plots  the second and third eigen vectors of the reweighted transition matrix
eig_vals_, eig_vecs_ = np.linalg.eig(reweighted_matrix.T)
left_eig_ = eig_vecs_.T
plt.plot(np.linspace(-2,2,40),left_eig_[1])
plt.xlabel('position coordinate along x')
plt.title('second MSM left eigen function')
plt.plot(left_eig[1])



"""
plots the boltzmann distribution and the simulated histogram
"""
from scipy.integrate import quad

instance1 = reweight(30, 8 , triple_well,double_well)

X1, eta1 = instance1.generate()
T_M1, _ = instance1.MSM(X1,10 )
pi1  = instance1.equilibrium_dist(T_M1)
plt.plot(pi1)
plt.hist(X1,bins =1000, density=True)


x = np.linspace(-2,2,200)
def potential_function(x):
    return np.exp(-4* (x**3 - (3/2) * x)**2 - x**3 + x)


boltzmann = np.exp(-triple_well.potential(x)/(kb*T))
Z , _ = quad(potential_function, -np.inf , np.inf)

plt.plot(x, boltzmann/Z , label ='Boltzmann distribution')
plt.plot(pi, label = 'reweighted')
plt.xlabel('position coordinate')
plt.ylabel('frequency density/ probability density')
plt.title('Expected equilibrium distribution, Double well potential')
plt.legend()


plt.hist(X1, bins =1000, density= True)




"""
plot the implied timescale  vs lagtimes for double well and triple well potentials 

"""
T_ = np.zeros((60,10,10))
T_1 = np.zeros_like(T_)
lag_times = np.linspace(1,120,60)
eig = np.zeros((60,10))
eig1 = np.zeros_like(eig)
eigen = np.zeros(60)
eigen1 = np.zeros_like(eigen)
cc = np.zeros(60)
cc1 = np.zeros_like(cc)
for i in range(len(lag_times)):

    T_[i], _ = instance.MSM(X,int(lag_times[i]))
    T_1[i], _ = instance.MSM(X1, int(lag_times[i]))
    eig[i] = np.linalg.eigvals(T_[i])
    eig1[i] = np.linalg.eigvals(T_1[i])
    eigen[i] = eig1[i][3]
    eigen1[i] = eig1[i][2]

    cc[i] = -lag_times[i]/np.log(eigen[i])
    cc1[i] = -lag_times[i] / np.log(eigen1[i])

plt.plot(lag_times,cc, label = 'double well')
plt.plot(lag_times, cc1, label = 'triple well')
plt.xlabel('lag times, 10^2 dt')
plt.ylabel('implied time scales, 10^2 dt')
plt.title('time scales vs lagtimes')
plt.legend()



"""
plots the simulated and target potentials
"""

plt.plot(np.linspace(-2,2,200), double_well.potential(np.linspace(-2,2,200)), label= 'doublewell potential')
plt.plot(np.linspace(-2,2,200), triple_well.potential(np.linspace(-2,2,200)), label= 'triplewell potential')
plt.xlabel('position coordinate')
plt.ylabel('potential V')
plt.ylim(top=10)
plt.ylim(bottom = -3)
plt.legend()

len = np.linspace(4,20,10)

def prob(len):

    return np.exp(-double_well.potential(X[0]))*np.prod(np.exp(-eta[:int(len)]**2/2))

probability = []
for i in len:

    probability.append(prob(i))


plt.plot(len, probability)
plt.xlabel('path length')
plt.ylabel('probability')
plt.title('path probabilities vs path length')

