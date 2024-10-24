from KineticsSandbox.system import system
import matplotlib.pyplot as plt
from KineticsSandbox.potential import D1
from underdamped import underdamped
import numpy as np

kb = 1
m = 1
x = 1.2
v = 1
T = 1
xi = 1
dt = 0.01
h = 0.01

system = system.D1(m,x,v,T,xi,dt,h)

doublewell = D1.DoubleWell([1,0,1])
triplewell = D1.TripleWell([4])

instance = underdamped(system,40,10,40, doublewell, triplewell)

X, v, eta1, delta_eta1 = instance.generate(int(1e7))

plt.hist(X, bins =1000, density=True)
plt.xlabel('coordinate along x')
plt.ylabel('frequency density')
plt.title('probability distribution (BAOAB)')

plt.hist(v, bins =1000, density=True)
plt.xlabel('coordinate along x ')
plt.ylabel('frequency density of velocities')
plt.title('maxwell boltzmann distribution (BAOAB)')

M = instance.reweighting_factor(X, eta1, delta_eta1)
reweighted_matrix = instance.reweighted_MSM(X, M)
reweighted_matrix = np.nan_to_num(reweighted_matrix,nan=0)
pi = instance.equilibrium_dist(reweighted_matrix)
plt.plot(pi)
plt.title('reweighted distribution (BAOAB)')


instance1 = underdamped(system,40,10,40, triplewell, doublewell)
X, v, eta1, delta_eta1 = instance1.generate(int(1e7))
T_M, path = instance1.MSM(X,10 )
eig_vals, eig_vecs = np.linalg.eig(T_M.T)
left_eig = eig_vecs.T

plt.plot(left_eig[1]/ np.sum(left_eig[1]), label = 'simulated eigen vector')
plt.plot(pi/np.sum(pi), label = 'reweighted eigen vector')
plt.title('reweighted distribution (BOAOB)')
plt.legend()

plt.hist(X, 1000,density=True)

