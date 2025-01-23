from KineticsSandbox.system import system
import matplotlib.pyplot as plt
from KineticsSandbox.potential import D1
from overdamped import reweight
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

doublewell = D1.DoubleWell([1,0,3])
triplewell = D1.TripleWell([4])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
reweighed_bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 2])

y = np.linspace(-6, 6, 200)
potential = bolhuis.potential(y)
potential_reweighted = reweighed_bolhuis.potential(y)
#plt.legend()

instance = reweight(system,100,200, triplewell, doublewell)
X, eta, delta_eta, times_added = instance.metadynamics_generate(int(1e6))

X, eta, delta_eta = instance.generate(int(1e6), times_added)
plt.hist(X, bins=1000)

fig,  (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.hist(X, bins =1000, density=True, label='direct simulation')
ax2.hist(X, bins =1000, density=True, label='metadynamics simulation')
ax1.legend()
ax2.legend()

plt.hist(times_added,bins=100)
plt.title('Gaussian positions')

plt.hist(X, bins =1000,density=True, label='Metadynamics simulation')
plt.legend()


transition_matrix = instance.MSM(X, 200)
transition_matrix = np.nan_to_num(transition_matrix, nan=0)
eq = instance.equilibrium_dist(transition_matrix)
plt.plot(eq)
plt.xlabel('coordinate along x')
plt.ylabel('frequency density')
plt.title('probability distribution (BAOAB)')

plt.hist(v, bins =1000, density=True)
plt.xlabel('coordinate along x ')
plt.ylabel('frequency density of velocities')
plt.title('maxwell boltzmann distribution (BAOAB)')

M = instance.reweighting_factor(X, eta, delta_eta,200, times_added)
reweighted_matrix = instance.reweighted_MSM(X, M, 200)
reweighted_matrix = np.nan_to_num(reweighted_matrix,nan=0)
pi = instance.equilibrium_dist(reweighted_matrix)
plt.plot(pi)
plt.title('reweighted distribution (BAOAB)')


x = np.linspace(min(X), max(X),100)
bins = np.linspace(min(X), max(X), 100 + 1, endpoint=True)
x_boltzmann = 0.5 * (bins[1:] + bins[:-1])
boltzmann = np.exp(-triplewell.potential(x_boltzmann)/(kb*T))
x_pi = 0.5 * (bins[1:] + bins[:-1])

plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution')
plt.plot(x_pi, pi/np.sum(pi), label = 'reweighted eigen vector')
plt.title('Overdamped EM scheme, alpha = 2')
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()


instance1 = reweight(system, 100, 1000, doublewell, triplewell)
X, eta, delta_eta = instance1.generate(int(1e7))

instance2 = reweight(system, 100, 1000, triplewell, doublewell)
X2, eta2, delta_eta2 = instance2.generate(int(1e7))

eig = np.zeros(100)
tau = np.linspace(1,800,8)

eigen1, eigen2, eigen3 = instance.implied_timescales(X,X2,eta, delta_eta, tau)

plt.plot(tau, eigen1, label='unbiased')
plt.plot(tau, eigen2, label ='reweighted')
plt.plot(tau ,eigen3, label= 'target', linestyle='--')
plt.xlabel('lagtimes (n steps)')
plt.ylabel('implied timescales, Overdamped')
plt.title('implied timescales v lagtimes')
plt.legend()


eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Extract the second and third eigenvectors
second_eigenvector = eigenvectors[:, 1]  # Index 1 corresponds to the second eigenvector
third_eigenvector = eigenvectors[:, 2]  # Index 2 corresponds to the third eigenvector

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Plot the eigenvectors
ax1.plot(-second_eigenvector, label='Second Eigenvector')
ax2.plot(third_eigenvector, label='Third Eigenvector')

# Set axis labels
ax1.set_xlabel('x coordinate')
ax2.set_xlabel('x coordinate')

# Set titles for each subplot
ax1.set_title('Second Eigenvector')
ax2.set_title('Third Eigenvector')

# Set the main title for the figure
fig.suptitle('Slowest Dynamical Processes')

# Show legends
ax1.legend()
ax2.legend()

# Display the figure
plt.tight_layout()
plt.show()




alpha = np.array([2,4,6,8,10,12])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
eigen3 = np.zeros((len(alpha), 9))
eigen2 = np.zeros((len(alpha), 9))
eigen1 = np.zeros((len(alpha), 9))

for i in range(len(alpha)):
    reweighed_bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, alpha[i]])
    instance = reweight(system, 100, 1000, bolhuis, reweighed_bolhuis)
    X, eta, delta_eta = instance.generate(int(5e6))
    instance2 = reweight(system, 100, 1000, reweighed_bolhuis, bolhuis)
    X2,_ ,_ = instance2.generate(int(5e6))
    lagtimes = np.linspace(10,1800,9)
    eigen1[i,:], eigen2[i,:], eigen3[i,:] = instance.implied_timescales(X, X2, eta, delta_eta, lagtimes)


np.save('eigen51_over.npy',eigen1)
np.save('eigen52_over.npy',eigen2)
np.save('eigen53_over.npy',eigen3)

lagtimes = np.linspace(10,1800,9)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,10))
ax1.plot(lagtimes, eigen3[0], label='alpha = 2', color = 'g')
ax1.plot(lagtimes, eigen2[0], label='alpha = 2', color ='g', linestyle = '--')
ax1.plot(lagtimes,eigen1[0], label='unbiased',color ='b')

ax1.plot(lagtimes, eigen3[1], label='alpha = 2', color = 'g')
ax1.plot(lagtimes, eigen2[1], label='alpha = 2', color ='g', linestyle = '--')

ax1.plot(lagtimes, eigen3[2], label = 'alpha = 4', color ='b')
ax1.plot(lagtimes, eigen2[2], color = 'b', linestyle = '--' )

ax2.plot(lagtimes, eigen3[3], label = 'alpha = 6', color ='b')
ax2.plot(lagtimes, eigen2[3], color = 'b', linestyle = '--' )

ax2.plot(lagtimes, eigen3[4], label= 'alpha =8', color = 'r')
ax2.plot(lagtimes, eigen2[4], color = 'r' )

ax2.plot(lagtimes, eigen3[5], label = 'alpha =10', color = 'g')
ax2.plot(lagtimes, eigen2[5], color = 'g', linestyle='--')

plt.xlabel('lagtimes')
plt.ylabel('implied timescales')
plt.title('Implied timescales absolute error')
ax1.legend()
ax2.legend()
plt.show()


def meta(x, times_added):
    x = np.array(x)

    bias = 0.0005 * np.sum(np.exp(- (x - np.array(times_added)) ** 2 / 0.2))

    return bias



x = np.linspace(-2,2,200)
bias = np.zeros_like(x)
for i, j in enumerate(x):
    bias[i] = meta(j,times_added)

plt.plot(x, bias)
plt.title('metadynamics bias')


def meta_force(x, times_added):
    x = np.array(x)
    force = 0.001 * np.sum((x - times_added) * np.exp(-(x - times_added)**2 / 0.2) / 0.01)
    return force

x = np.linspace(-2,2,200)
bias = np.zeros_like(x)
for i, j in enumerate(x):
    bias[i] = meta_force(j,times_added)

plt.plot(x, bias)


plt.plot(x, triplewell.potential(x), label='unbiased')
plt.plot(x, triplewell.potential(x) + bias, label = 'biased')
plt.legend()


x_pos = np.linspace(-2,2,20000)
def calculate_force(x, times_added, sigma):
    return 0.0005 * np.sum((x - times_added) * np.exp(-(x - times_added)**2 / 0.2) / 0.01)

y_true = np.array([calculate_force(x, times_added, 0.005) for x in x_pos])
y_true1 = y_true + triplewell.force_ana(x_pos)[0]

plt.plot(x_pos , y_true1)


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_true = scaler.fit_transform(y_true.reshape(-1,1))
x_train = tf.convert_to_tensor(x_pos.reshape(-1, 1), dtype=tf.float32)
y_train = tf.convert_to_tensor(y_true, dtype=tf.float32)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),  # Hidden layer 1
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2
    tf.keras.layers.Dense(1)  # Output layer for the force
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32)


x_test = np.linspace(-2, 2, 300).reshape(-1, 1)

# Predict the force for the new array of positions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

# Calculate the actual force using the same formula
y_actual = np.array([calculate_force(x, times_added, 0.005) for x in x_test])

# Plot the actual vs predicted forces
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_actual, label='Actual Force', color='blue')
plt.plot(x_test, y_pred, label='Predicted Force', color='orange', linestyle='dashed')
plt.xlabel('Position (x)')
plt.ylabel('Force')
plt.title('Actual vs Predicted Force')
plt.legend()
plt.show()

model.save('force_prediction_model.h5')

plt.plot(doublewell.force_ana(y)[0])


