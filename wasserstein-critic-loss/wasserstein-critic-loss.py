import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    
    real_scores : critic outputs for real samples
    fake_scores : critic outputs for fake samples
    """

    real_scores = np.asarray(real_scores, dtype=float)
    fake_scores = np.asarray(fake_scores, dtype=float)

    # Compute means
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)

    # Wasserstein critic loss
    loss = fake_mean - real_mean

    return loss