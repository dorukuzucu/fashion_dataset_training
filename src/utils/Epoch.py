import time


class Epoch:
    def __init__(self):
        self.id = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = 0
        self.train_data_len = 0

    def end(self):
        self.id = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = 0

    def begin(self):
        self.start_time = time.time()
        self.id += 1
