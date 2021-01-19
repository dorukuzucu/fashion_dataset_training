import time


class RunUtil:

    def __init__(self):
        # for metrics
        self.id = 0
        self.params = 0
        self.data = 0
        self.start_time = 0

        # for training
        self.lr = 0
        self.epochs = 0
        self.batch_size = 0
        self.num_workers = 0
        self.device = None

    def begin(self, run):
        self.id += 1
        self.lr = run.lr
        self.epochs = run.epochs
        self.batch_size = run.batch_size
        self.num_workers = run.num_workers
        self.device = run.working_on
        self.start_time = time.time()

    def end(self):
        # reset metrics
        self.params = 0
        self.data = 0
        self.start_time = 0

        # for training
        self.lr = 0
        self.epochs = 0
        self.batch_size = 0
        self.num_workers = 0
        self.device = None
