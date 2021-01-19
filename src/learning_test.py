from src.model.Net import Net
from collections import OrderedDict
from src.model.Trainer import Trainer
from src.utils.RunBuilder import RunBuilder
from src.data.FashionMNISTDataLoader import FashionMNISTDataLoader


hyper_params = OrderedDict(
    lr=[0.001, 0.003, 0.005],
    batch_size=[100, 500, 1000, 5000, 10000],
    epochs=[3, 4, 5],
    num_workers=[0],
    working_on=['cuda']
)

# learning runs
run_builder = RunBuilder()
run_builder.add_runs(params=hyper_params)
runs = run_builder.get_runs()

#neural net and data_loader
net = Net()
dataset_loader = FashionMNISTDataLoader()
train_mng = Trainer(net, dataset_loader)

# train for different runs
train_mng.train(runs)
# save results of training
train_mng.save_results("test_run")
