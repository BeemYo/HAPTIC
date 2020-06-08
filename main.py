import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.2, 0.8, -0.5],
           [0.5, -0.91, 0.26],
           [-0.26, -0.27, 0.17]]

biases2 = [3, 1, 0.5]

output_1 = np.dot(inputs, np.array(weights).T) + biases
output_2 = np.dot(output_1, np.array(weights2).T) + biases2
print(output_1)