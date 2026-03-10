import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # Handle (D,) input → convert to (1, D)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Validate y values
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)

    # Contrastive loss
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2

    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss