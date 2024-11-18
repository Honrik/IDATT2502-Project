class PPOHyperparameters:
    def __init__(self,
                 gamma = 0.99,
                 gae_lambda = 0.96,
                 alpha = 0.0003,
                 policy_clip=0.2,
                 entropy_coeff = 0.01,
                 horizon = 2048,
                 minibatch_size = 32,
                 n_epochs = 10
                 ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.policy_clip = policy_clip
        self.entropy_coeff = entropy_coeff
        self.horizon = horizon
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs