import json
import time

import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

from src.utils.Epoch import Epoch
from collections import OrderedDict
from src.utils.RunUtil import RunUtil
from torch.utils.tensorboard import SummaryWriter
from src.data.FashionMNISTDataLoader import FashionMNISTDataLoader


# TODO add train set normalization parameter to data_loader and train_manager
class Trainer:
    def __init__(self, net: torch.nn.Module, data_loader: FashionMNISTDataLoader):
        # set data_loader and neural net
        self.neural_net = net
        self.data_loader = data_loader
        # run&epoch classes
        self.epoch = Epoch()
        self.run = RunUtil()
        # save summaries for tensorboard
        self.summary_writer = None
        self.run_data = []
        # training configs
        torch.set_grad_enabled(True)
        torch.set_printoptions(linewidth=80)

    def train(self, runs):
        for run_config in runs:
            # assign configs to variable
            self.run_config = run_config

            # set parameters for this run
            self.__begin_run(self.run_config)

            # convert nn params to device if available
            self.neural_net.to(self.run.device) if self.__is_cuda_enabled() else print("CUDA can not be used")

            # set optimizer
            self.__set_optimizer()

            # get train data
            train_data_loader, self.epoch.train_data_len = self.data_loader.get_train_data_loader(self.run.batch_size,
                                                                                      self.run.num_workers)

            # define summary writer for tensorboard
            self.summary_writer = SummaryWriter(comment=f'-{self.run_config}')

            # training epochs
            for epoch in range(self.run.epochs):
                #begin epoch
                self.__begin_epoch()

                # iterate over images
                for batch in train_data_loader:
                    # check if cuda is enabled
                    if self.__is_cuda_enabled():
                        images = batch[0].to(self.run.device)
                        labels = batch[1].to(self.run.device)
                    else:
                        images, labels = batch
                    preds = self.neural_net(images)
                    loss = F.cross_entropy(preds, labels)  # calculate the loss

                    # zero out gradient values
                    self.optimizer.zero_grad()
                    loss.backward()  # calculate the gradients
                    self.optimizer.step()  # update weights of network

                    # calculate metrics
                    self.epoch.loss += loss.item() * self.run.batch_size
                    self.epoch.num_correct += preds.argmax(dim=1).eq(labels).sum().item()

                print("run:", self.run.id, "epoch:", epoch, "total_correct:", self.epoch.num_correct, "loss:",
                      self.epoch.loss)
                self.__end_epoch()
            self.__end_run()

    def __end_epoch(self):
        self.__save_epoch_metrics()
        self.epoch.end()

    def __end_run(self):
        self.run.end()
        self.summary_writer.close()

    def __begin_run(self, run_cfg):
        self.run.begin(run_cfg)

    def __begin_epoch(self):
        self.epoch.begin()

    def __set_optimizer(self):
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=self.run.lr)

    def __save_epoch_metrics(self):
        # add scalar values
        self.accuracy = self.epoch.num_correct / self.epoch.train_data_len
        self.summary_writer.add_scalar("Loss", self.epoch.loss, self.epoch.id)
        self.summary_writer.add_scalar("Accuracy", self.accuracy, self.epoch.id)

        # save network parameters
        for name, weight in self.neural_net.named_parameters():
            self.summary_writer.add_histogram(name, weight, self.epoch.id)
            self.summary_writer.add_histogram(f'{name}.grad', weight.grad, self.epoch.id)

        # save metrics to results
        self.run_data.append(self.__save_to_results())

    def __save_to_results(self):
        #add results to an array
        results = OrderedDict()
        results["run"] = self.run.id
        results["epoch"] = self.epoch.id
        results["loss"] = self.epoch.loss
        results["accuracy"] = self.accuracy
        results["epoch duration"] = time.time() - self.epoch.start_time
        for key, value in self.run_config._asdict().items(): results[key] = value
        return results

    # check if cuda is enabled in both device and run
    def __is_cuda_enabled(self):
        if self.run.device == 'cuda' and torch.cuda.is_available():
            return True
        else:
            return False

    # return trained network
    def trained_net(self):
        return self.neural_net

    def save_results(self, filename: str):
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)