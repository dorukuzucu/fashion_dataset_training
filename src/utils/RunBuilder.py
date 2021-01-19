from collections import OrderedDict
from collections import namedtuple
from itertools import product


class RunBuilder():
    def __init__(self):
        self.runs = []

    def add_runs(self, params: OrderedDict):
        # get keys and values
        self.keys = params.keys()
        self.values = params.values()

        # add them to array
        Run = namedtuple('Run', self.keys)
        for value in product(*self.values):
            self.runs.append(Run(*value))

    # return runs
    def get_runs(self):
        return self.runs
