from KineticsSandbox.system import system
import matplotlib.pyplot as plt
from KineticsSandbox.potential import D1
from underdamped import underdamped
import numpy as np
from scipy import constants

kb = constants.R * 0.001
m = 1
x = 1.5
v = 0
T = 300
xi = 50
dt = 0.01
h = 0.01

system = system.D1(m,x,v,T,xi,dt,h)

doublewell = D1.DoubleWell([1,0,1])
triplewell = D1.TripleWell([4])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
reweighed_bolhuis  = D1.Bolhuis([0, 0, 1, 1, 1, 5])

y = np.linspace(-2, 2, 200)
potential = bolhuis.potential(y)
potential_reweighted = reweighed_bolhuis.potential(y)

instance = underdamped(system,100,400, doublewell, triplewell)

X, v, eta1, delta_eta1 = instance.generate(int(1e7))
plt.hist(X, bins =1000)

plt.hist(X, bins =1000, density=True)

transition_matrix = instance.MSM(X, 20)
transition_matrix = np.nan_to_num(transition_matrix, nan=0)
eq = instance.equilibrium_dist(transition_matrix)
plt.plot(eq)
plt.xlabel('coordinate along x')
plt.ylabel('frequency density')
plt.title('probability distribution (ABOBA)')

plt.hist(v, bins =1000, density=True)
plt.xlabel('coordinate along x ')
plt.ylabel('frequency density of velocities')
plt.title('maxwell boltzmann distribution (ABOBA)')

M = instance.reweighting_factor(X, eta1, delta_eta1, 400)
reweighted_matrix = instance.reweighted_MSM(X, M,400)
reweighted_matrix = np.nan_to_num(reweighted_matrix, nan=0)
pi = instance.equilibrium_dist(reweighted_matrix)
plt.plot(pi)
plt.title('reweighted distribution (ABOBA)')


"""



implied time scales plot

"""


instance1 = underdamped(system, 10, 1000, triplewell, doublewell)
X1, _, eta1, delta_eta1 = instance1.generate(int(4e6))

instance2 = underdamped(system,  10, 1000, doublewell, triplewell)
X2, _, eta2, delta_eta2 = instance2.generate(int(4e6))

eigen1 = np.zeros(12)
eigen2 = np.zeros(12)
eigen3 = np.zeros(12)
eig = np.zeros(10)
tau = np.linspace(10,1000,12)


for i in range(12):

    T_ = instance2.MSM(X2, int(tau[i]))
    T_ = np.nan_to_num(T_, nan=0)
    eig = np.linalg.eigvals(T_)
    eig = np.sort(eig)
    print(eig[-2])
    eigen1[i] = -tau[i] / np.log(eig[-2])

    M = instance2.reweighting_factor(X2, eta2, delta_eta2, lagtime=int(tau[i]))
    Tr = instance2.reweighted_MSM(X2, M, int(tau[i]))
    Tr = np.nan_to_num(Tr, nan=0)
    eig = np.linalg.eigvals(Tr)
    eig = np.sort(eig)
    print(eig[-2])
    eigen2[i] = -tau[i] / np.log(eig[-2])

    T_ = instance2.MSM(X1, int(tau[i]))
    T_ = np.nan_to_num(T_, nan=0)
    eig = np.linalg.eigvals(T_)
    eig = np.sort(eig)
    print(eig[-2])
    eigen3[i] = -tau[i] / np.log(eig[-2])


plt.plot(tau, eigen1, label='doublewell')
plt.plot(tau, eigen2, label ='reweighted eigenvector')
plt.plot(tau ,eigen3, label= 'simulated triplewell')
plt.xlabel('lagtimes (n steps)')
plt.ylabel('implied timescales')
plt.title('implied timescales v lagtimes')
plt.legend()


instance1 = underdamped(system,100,200, triplewell, doublewell)
X1, v, eta1, delta_eta1, = instance1.generate(int(5e6))

TM = instance1.MSM(X1,20)
TM = np.nan_to_num(TM,nan=0)
pi1 = instance1.equilibrium_dist(TM)
plt.plot(pi1)


"""analytical boltzmann distribution vs first eigen vectors"""


x = np.linspace(min(X) - 1e-6, max(X) + 1e-6,100)

boltzmann = np.exp(-triplewell.potential(x)/(kb*T))

x_boltzmann = np.linspace(min(X) - 1e-6,max(X) + 1e-6,len(boltzmann))
x_pi = np.linspace(min(X) - 1e-6, max(X) + 1e-6,len(pi))

#plt.plot(x_pi,pi1/np.sum(pi1), label='simulated eigen vector')
plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution')
plt.plot(x_pi, pi/np.sum(pi), label = 'reweighted eigen vector')
plt.title('underdamped ABOBA , alpha = 20')
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()


bins = np.linspace(min(X) - 1e-6, max(X) + 1e-6, 11)
plt.plot(bins)
discretized_path = np.digitize(X, bins, right=False)
max(discretized_path)

plt.hist(X, bins=1000)







