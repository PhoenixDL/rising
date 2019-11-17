#####################
#       Logging     #
#####################
import logging
import sys
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Execute_Logger")

#########################
#       Training        #
#########################
import os
import torch
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from batchgenerators.transforms import Compose, MirrorTransform, \
    SpatialTransform, ZeroMeanUnitVarianceTransform

import delira
from delira.training import Parameters
from delira.data_loading import BaseDataManager, SequentialSampler, \
    RandomSampler
from delira.training.train_utils import create_optims_default_pytorch

# Template Imports
from delira.data_loading import TorchvisionClassificationDataset
from delira.models.classification import ClassificationNetworkBasePyTorch
from delira.training import PyTorchExperiment

#############
#   Params  #
#############
# name of current experiment
exp_name = "TemplateExperiment"
# path where experiment will be saved (including models)
save_path = "path/to/experiment"
checkpoint_freq = 1

# path to training data
training_path = "path/to/training_data"
# path to test data
test_path = "path/to/test_data"
# precentiele which is split from training set to validate results
val_size = 0.2
# seed for split
seed = 0
val_score_key = "val_score_key"
val_score_mode = "val_score_mode"
n_process_augmentation = ch['n_process_augmentation']

# Parameters
losses = {"CE": torch.nn.CrossEntropyLoss()}
params = Parameters(fixed_params={
    "model": {
        "in_channels": 1,
        "n_outputs": 10
    },
    "training": {
        "batch_size": 64,  # batchsize to use
        "num_epochs": 10,  # number of epochs to train
        "optimizer_cls": torch.optim.Adam,  # optimization algorithm to use
        # initialization parameters for this algorithm
        "optimizer_params": {'lr': 1e-3},
        "losses": losses,  # the loss function
        "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
        "lr_sched_params": {},  # the corresponding initialization parameters
        "metrics": {}  # and some evaluation metrics
    }
})

#################
#   Datasets    #
#################
dset = TorchvisionClassificationDataset("mnist",  # which dataset to use
                                        train=True,  # use trainset
                                        # resample to 224 x 224 pixels
                                        img_shape=(224, 224),
                                        )
idx = list(range(dset))
idx_train, idx_val = train_test_split(
    idx, test_size=val_size, random_state=seed)

dataset_train = dset.get_subset(idx_train)
dataset_val = dset.get_subset(idx_val)

dataset_test = TorchvisionClassificationDataset("mnist",
                                                train=False,
                                                img_shape=(224, 224),
                                                )

#####################
#   Augmentation    #
#####################
base_transforms = [ZeroMeanUnitVarianceTransform(),
                   ]
train_transforms = [SpatialTransform(patch_size=(200, 200),
                                     random_crop=False,
                                     ),
                    ]

#####################
#   Datamanagers    #
#####################
manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                transforms=Compose(
                                    base_transforms + train_transforms),
                                sampler_cls=RandomSampler,
                                n_process_augmentation=n_process_augmentation)

manager_val = BaseDataManager(dataset_val, params.nested_get("batch_size"),
                              transforms=Compose(base_transforms),
                              sampler_cls=SequentialSampler,
                              n_process_augmentation=n_process_augmentation)

manager_test = BaseDataManager(dataset_test, 1,
                               transforms=Compose(base_transforms),
                               sampler_cls=SequentialSampler,
                               n_process_augmentation=n_process_augmentation)

logger.info("Init Experiment")
experiment = PyTorchExperiment(params,
                               ClassificationNetworkBasePyTorch,
                               name=exp_name,
                               save_path=save_path,
                               checkpoint_freq=checkpoint_freq,
                               optim_builder=create_optims_default_pytorch,
                               gpu_ids=[0],
                               mixed_precision=False,
                               val_score_key=val_score_key,
                               )
experiment.save()
net = experiment.run(manager_train, manager_val,
                     val_score_mode=val_score_mode,
                     verbose=True,
                     )

experiment.test(net, manager_test,
                verbose=True,
                )
