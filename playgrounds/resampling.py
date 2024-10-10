def normalize_weights(self):
    """
    Normalizes the weights of all particles.
    """
    total_weight = sum(p.weight for p in self.particles)
    if total_weight == 0:
        for p in self.particles:
            p.weight = 1.0 / self.num_particles
    else:
        for p in self.particles:
            p.weight /= total_weight


def asaptive_resampling(self):
    """
    This method is based on the number of effective particles.
    The idea is that resampling is performed when the effective number of particles is below a certain threshold (in this case half of the total number of particles).
    """
    effective_particles = 1 / sum(p.weight ** 2 for p in self.particles)
    return effective_particles < self.num_particles / 2


def weight_unbalanced(self):
    """
    Checks if the particle weights are unbalanced, indicating resampling is needed.

    :return: True if resampling is needed, False otherwise.

    This method checks the imbalance of the particle weights based on the variance.
    Resampling is performed when the variance exceeds a certain threshold value.
    """
    variance = sum((p.weight - 1.0 / self.num_particles) ** 2 for p in self.particles)
    threshold = ((self.num_particles - 1) / self.num_particles) ** 2 + \
                (self.num_particles - 1) * (1 / self.num_particles) ** 2
    return variance > threshold


def resample(self):
    """
    Resamples particles based on their weights.
    """
    weights = [p.weight for p in self.particles]

    if np.sum(weights) == 0:
        weights = [1.0 / self.num_particles] * self.num_particles

    resampled_indices = np.random.choice(range(self.num_particles), self.num_particles, p=weights)
    self.particles = [copy.deepcopy(self.particles[i]) for i in resampled_indices]

    for p in self.particles:
        p.weight = 1.0 / self.num_particles