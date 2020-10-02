import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_pd = pd.read_csv('ex1data1.csv',header = None)
training_size = 90
alpha = 0.008
data_np = data_pd.to_numpy()
# profits = []
# population = []
# for data in data_np:
#     profits.append(data[1])
#     population.append(data[0])

output_test = data_np[90:, 0]
output_training = data_np[:90, 0]
output_test = np.reshape(output_test, (len(output_test), 1))
output_training = np.reshape(output_training, (len(output_training), 1))
plt.plot(data_np[:, 0] ,data_np[:, 1], 'rx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profits in $10,000')

m = training_size
data_train = np.reshape(data_np[:m, 1], (m, 1))
data_test = np.reshape(sorted(data_np[m:, 1]), (data_np.shape[0] - m,1))

full_data = np.hstack((np.ones((m, 1)), data_train))
print(np.ones((data_np.shape[0] - m, 1)).shape)
print(data_test.shape)
full_data_test = np.hstack((np.ones((data_np.shape[0] - m, 1)), data_test))
prams = np.random.random((full_data.shape[1], 1))

training_data = full_data[:training_size,:]
test_Data = full_data[training_size:, :]



def gradient_descent(prams, training_data, output_training):
    update = ((alpha/(training_size)) * ((np.matmul(np.transpose(training_data),np.subtract(np.matmul(training_data, prams),output_training ) ))))

    prams = prams - update
    return prams
for iter in range(1000):

    prams = gradient_descent(prams, training_data, output_training)

current_error = np.sum(np.square(np.subtract(np.matmul(full_data_test, test_prams),output_test ))) / (2 * 10)
print(current_error)
x = np.arange(5.0, 22.5, 0.5)
x = np.reshape(x, (35,1))
input_x = np.hstack((np.ones((35,1)),x ))

test_prams = np.array([[-4.5], [1.25]])
print(prams)

plt.plot(x, np.matmul(input_x, test_prams))
plt.show()




