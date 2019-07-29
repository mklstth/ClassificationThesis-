
import matplotlib.pyplot as plt
import numpy as np


# with open('matrix.txt') as f:
#     matrix = float(f)


matrix = np.random.random((8, 8))
plt.imshow(matrix, interpolation='nearest')
plt.title('Confusion matrix')
plt.yticks(np.arange(8),('Chest', 'Elbow', 'Finger', 'Forearm', 'Hand', "Humerus", 'Shoulder', 'Wrist'))
plt.xticks(np.arange(8),('Chest', 'Elbow', 'Finger', 'Forearm', 'Hand', "Humerus", 'Shoulder', 'Wrist'), rotation=45)
plt.show()