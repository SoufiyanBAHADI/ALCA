class LearningManager:
    def __init__(self, optimizer, buffer_size, epochs, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.lr = optimizer.defaults['lr']
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.epochs = epochs

