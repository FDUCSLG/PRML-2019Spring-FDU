import numpy as np

def temporal_softmax_loss(x, y, mask, tau=1):
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp((x_flat - np.max(x_flat, axis=1, keepdims=True)) / tau)
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

def softmax(x, tau=1):
    probs = np.exp((x - np.max(x, axis=-1, keepdims=True)) / tau)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    return np.max(probs, axis=-1, keepdims=True)