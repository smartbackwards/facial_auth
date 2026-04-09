import numpy as np

def compute_FAR(scores, labels, threshold):
    impostors = scores[labels == 0]
    return np.mean(impostors >= threshold)

def compute_FRR(scores, labels, threshold):
    genuines = scores[labels == 1]
    return np.mean(genuines < threshold)

def threshold_sweep(scores, labels, steps=500):
    thresholds = np.linspace(0, 1, steps)
    fars = [compute_FAR(scores, labels, t) for t in thresholds]
    frrs = [compute_FRR(scores, labels, t) for t in thresholds]
    return np.array(thresholds), np.array(fars), np.array(frrs)