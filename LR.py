import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


training_size = 90
alpha = 0.008
lables = []
features = []



def data_read(name):
    global lables
    global features
    global training_size

    data_pd = pd.read_csv(name,header = None)

    data_np = data_pd.to_numpy()

    m = data_np.shape[0]
    n = data_np.shape[1] - 1

    lables = np.reshape(data_np[:, 0], (m, 1))
    features = np.reshape(data_np[:, 1:], (m , n))

    training_lables= lables[:training_size]
    testing_labels = lables[training_size:]

    training_data = np.hstack((np.ones((training_size, 1)), features[:training_size]))
    testing_data = np.hstack((np.ones((m - training_size, 1)), features[training_size:]))

    return training_lables, training_data, testing_labels, testing_data, m, n


def init_prams(n):

    prams = np.random.random((n+1, 1))
    return prams



def gradient_descent(prams, training_data, output_training):

    update = ((alpha/(training_size)) * ((np.matmul(np.transpose(training_data),np.subtract(np.matmul(training_data, prams),output_training ) ))))

    prams = prams - update
    return prams


def gradientDescentMulti(X, y, theta, alpha, iter):

    J_history = []

    m = len(y)
    for i in range(iter):
        h = X.dot(theta)
        theta = theta - (alpha / m) * (X.T.dot (h - y))
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

#Nomalizing features
def featureNormalize(X):
    X_norm=(X-np.mean(X))/np.std(X)
    mu = np.mean(X)
    sigma = np.std(X)
    return X_norm, mu, sigma


def computeCostMulti(X,y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1/(2*m)*(np.sum((h-y)**2))
    return J

def prediction(X, theta):
    return np.dot(X, theta)

y_train, x_train, y_test, x_test, m, n = data_read("ex1data1.csv")

prams = init_prams(n)
plt.plot(lables ,features, 'rx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profits in $10,000')

for iter in range(10000):

    prams = gradient_descent(prams, x_train, y_train)


test_prams = np.array([[-4.5], [1.25]])

current_error = np.sum(np.square(np.subtract(np.matmul(x_train, prams),y_train))) / training_size

print(current_error)

x = np.arange(5.0, 22.5, 0.5)
x = np.reshape(x, (35,1))
input_x = np.hstack((np.ones((35,1)),x ))


print(prams)
plt.plot(x, np.matmul(input_x, prams))
plt.show()

#Part-II with multiple variables
#importing Data
print('Part2')
data2 = pd.read_csv('ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])

x= data2.drop(['Price'],axis=1)

y=data2['Price']

m=len(y)

X, mean, std =featureNormalize(x)

X=np.append(np.ones([m,1]),X,axis=1)

y = np.array(y).reshape(-1,1)

theta = np.zeros([3,1])

cost = computeCostMulti(X,y,theta)

print(cost)

iter = 500
alpha = 0.01

new_theta, J_history = gradientDescentMulti(X, y, theta, alpha, iter)
print (new_theta)

new_cost = computeCostMulti(X,y,new_theta)
print(new_cost)

plt.plot(J_history)
plt.ylabel('Cost J')
plt.xlabel('Number of Iterations')
plt.title('Minimizing Cost Using Gradient Descent')
plt.show()


theta_1, J_history_1 = gradientDescentMulti(X, y, theta, 0.3, 50)
theta_2, J_history_2 = gradientDescentMulti(X, y, theta, 0.1, 50)
theta_3, J_history_3 = gradientDescentMulti(X, y, theta, 0.03, 50)
theta_4, J_history_4 = gradientDescentMulti(X, y, theta, 0.01, 50)
theta_5, J_history_5 = gradientDescentMulti(X, y, theta, 0.003, 50)
theta_6, J_history_6 = gradientDescentMulti(X, y, theta, 0.001, 50)

plt.plot(J_history_1, label='0.3')
plt.plot(J_history_2, label='0.1')
plt.plot(J_history_3, label='0.03')
plt.plot(J_history_4, label='0.01')
plt.plot(J_history_5, label='0.003')
plt.plot(J_history_6, label='0.001')
plt.title('Testing Different Learning Rates')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost J')
plt.show()

theta, J_history = gradientDescentMulti(X, y, theta, 0.3, 1500)
print(theta)
X = np.array([1650, 3])
X = (X - mean)/std
X = np.append(1, X)
X = np.reshape(X, (1,3))
print(X)
pred = prediction(X, theta)
print(pred)

#Part 3
print('Part 3')

X = np.append(np.ones([m,1]), x, axis=1)

theta = np.linalg.inv((X.T.dot(X))).dot(X.T.dot(y))
print(theta)

theta=theta.reshape(3,1)
print(theta)
X = [1650, 3]

X_new = np.append(1, X)
print(X_new)

Pred = prediction(X_new, theta)
print(Pred)