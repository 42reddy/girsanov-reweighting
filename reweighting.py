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
reweighed_bolhuis  = D1.Bolhuis([0, 0, 1, 1, 1, 4])

y = np.linspace(-2, 2, 200)
potential = bolhuis.potential(y)
potential_reweighted = reweighed_bolhuis.potential(y)
#plt.plot(y, potential, label = 'alpha = 0 ')
#plt.plot(y, potential_reweighted, label = 'alpha = 4')
#plt.legend()

instance = underdamped(system,100,300, bolhuis, reweighed_bolhuis)

X, v, eta1, delta_eta1= instance.generate(int(5e6))

plt.hist(X, bins =1000, density=True)

transition_matrix = instance.MSM(X, 300)
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

M = instance.reweighting_factor(X, eta1, delta_eta1)
reweighted_matrix = instance.reweighted_MSM(X, M)
reweighted_matrix = np.nan_to_num(reweighted_matrix, nan=0)
pi = instance.equilibrium_dist(reweighted_matrix)
plt.plot(pi)
plt.title('reweighted distribution (ABOBA)')


"""



implied time scales plot

"""


instance1 = underdamped(system, 10, 20, bolhuis, doublewell)
X1, _, eta1, delta_eta1 = instance1.generate(int(1e6))

eigen1 = np.zeros(10)
eigen2 = np.zeros(10)
eigen3 = np.zeros(10)
eig = np.zeros(100)
tau = np.linspace(10,900,10)


for i in range(10):

    T_ = instance1.MSM(X1, int(tau[i]))
    T_ = np.nan_to_num(T_, nan=0)
    eig = np.linalg.eigvals(T_)
    eig = np.sort(eig)
    print(eig[-2])
    eigen1[i] = -tau[i] / np.log(eig[-2])

    instance2 = underdamped(system, 10, int(tau[i] + 5), doublewell, triplewell)
    X1, _ , eta1, delta_eta1 = instance2.generate(int(1e6))
    M1 = instance2.reweighting_factor(X1, eta1, delta_eta1)
    Tr = instance2.reweighted_MSM(X1, M, int(tau[i]))
    Tr = np.nan_to_num(Tr, nan=0)
    eig = np.linalg.eigvals(Tr)
    eig = np.sort(eig)
    print(eig[-2])
    eigen2[i] = -tau[i] / np.log(eig[-2])


plt.plot(tau, eigen1, label='triplewell')
plt.plot(tau, eigen2, label ='reweighted triplewell')
plt.xlabel('lagtimes')
plt.ylabel('implied timescales')
plt.legend()


instance1 = underdamped(system,100,200, triplewell, doublewell)
X1, v, eta1, delta_eta1, = instance1.generate(int(5e6))

TM = instance1.MSM(X1, 200)
TM = np.nan_to_num(TM,nan=0)
pi1 = instance1.equilibrium_dist(TM)
plt.plot(pi1)


"""analytical boltzmann distribution vs first eigen vectors"""


x = np.linspace(-2,2,100)

boltzmann = np.exp(-reweighed_bolhuis.potential(x)/(kb*T))

x_boltzmann = np.linspace(-1.7,1.6,len(boltzmann))
x_pi = np.linspace(-1.7,1.6,len(pi))

#plt.plot(x_pi,pi1/np.sum(pi1), label='simulated eigen vector')
plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution')
plt.plot(x_pi, pi/np.sum(pi), label = 'reweighted eigen vector')
plt.title('underdamped ABOBA , lagtime = 200')
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()

