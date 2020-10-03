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




