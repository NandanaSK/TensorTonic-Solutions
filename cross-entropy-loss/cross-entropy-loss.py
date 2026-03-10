import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    y_true : list or array of true class labels (e.g., [0,2,1])
    y_pred : 2D array of predicted probabilities
             each row corresponds to a sample
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = len(y_true)   # number of samples
    loss = 0

    for i in range(N):
        # probability of the correct class
        p = y_pred[i][y_true[i]]
        loss += -np.log(p)

    return loss / N


# Example
y_true = [0, 2, 1]

y_pred = [
    [0.7, 0.2, 0.1],
    [0.1, 0.3, 0.6],
    [0.2, 0.5, 0.3]
]

loss = cross_entropy_loss(y_true, y_pred)
print("Cross Entropy Loss:", loss)