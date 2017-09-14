"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os
import pickle
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

import models.multitask_classifier as mymodel
import models.multitask_regressor as mymodel2
import models.layers as mylayer
import models.graph_models as mysequence

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--decay_epoch', default=30, type=int)# 262*15
parser.add_argument('--nb_epoch', default=27, type=int)
parser.add_argument('--gpu', default=2, type=int)
parser.add_argument('--yita', default=0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--graph_hid_num', default=2, type=int)
parser.add_argument('--snapshot', default='', type=str)
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--gcn', default=0, type=int)
parser.add_argument('--dataset', default='tox21', type=str)
args = parser.parse_args()

regress = 0
if args.dataset == 'tox21':
    from examples.tox21.tox21_datasets import load_tox21 as load_data
elif args.dataset == 'hiv':
    from examples.hiv.hiv_datasets import load_hiv as load_data
elif args.dataset == 'toxcast':
    from examples.toxcast.toxcast_datasets import load_toxcast as load_data
elif args.dataset == 'muv':
    from examples.muv.muv_datasets import load_muv as load_data
elif args.dataset == 'pcba':
    from examples.pcba.pcba_datasets import load_pcba as load_data
elif args.dataset == 'sampl':
    from examples.sampl.sampl_datasets import load_sampl as load_data
    regress = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if args.split == 0:
    split = 'index'
elif args.split == 1:
    split = 'random'
else:
    split = 'scaffold'

snapshot_path = 'examples/' + args.dataset +'/snapshot/'
data_path = 'examples/' + args.dataset +'/data/'
if not os.path.exists(snapshot_path):
    os.mkdir(snapshot_path)
if not os.path.exists(data_path):
    os.mkdir(data_path)
    
logdir = snapshot_path + split + '_yita_' + str(args.yita) + args.snapshot
if args.gcn==1:
    logdir = logdir + '_gcn'
f = open(logdir + '.log','w')
f.write(str(args) + '\n')
print(args)
f.close()    
    
# Load dataset
n_features = 75
dataset_path = data_path + split + '_datasets.pkl'
trans_path = data_path + split + '_trans.pkl'
tasks_path = data_path +'tasks.pkl'

if os.path.exists(data_path) and os.path.exists(trans_path) and os.path.exists(tasks_path): 
    with open(tasks_path, 'r') as f:
        tasks = pickle.load(f)
    with open(dataset_path, 'r') as f:
        datasets = pickle.load(f)
    with open(trans_path, 'r') as f:
        transformers = pickle.load(f)
else:
    tasks, datasets, transformers = load_data(featurizer='GraphConv', split=split)
    with open(tasks_path, 'w') as f:
        pickle.dump(tasks, f)
    with open(dataset_path, 'w') as f:
        pickle.dump(datasets, f)
    with open(trans_path, 'w') as f:
        pickle.dump(transformers, f)

train_dataset, valid_dataset, test_dataset = datasets
x_shape, y_shape, w_shape, i_shape = train_dataset.get_shape()
batch_size = args.batch_size
train_step = int(x_shape[0] / batch_size)
decay_step = train_step * args.decay_epoch 
# Fit models
if regress == 1:
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
else:
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)



graph_model = mysequence.SequentialGraph(n_features)
if args.gcn==1:
    graph_model.add(dc.nn.GraphConv(128, n_features, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())
    graph_model.add(dc.nn.GraphConv(128, 128, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())
else:
    graph_model.add(mylayer.GraphResBlock(128, n_features,batch_size, activation='relu',epsilon=1e-5,mode=1,bn_first=False,add=False,weight_decay=args.weight_decay))
    graph_model.add(mylayer.GraphPool(batch_size))
    for _ in xrange(args.graph_hid_num):
        graph_model.add(mylayer.GraphResBlock(128, 128, batch_size, activation='relu',epsilon=1e-5,mode=1,bn_first=False,add=True,weight_decay=args.weight_decay))
#        graph_model.add(mylayer.GraphResBlock(128, 128, activation='relu',epsilon=1e-5,mode=1,bn_first=False,add=True,weight_decay=args.weight_decay))
        graph_model.add(mylayer.GraphPool(batch_size))
# Gather Projection
graph_model.add(dc.nn.Dense(256, 128, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
if args.gcn==1:
    graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
else:
    graph_model.add(mylayer.GraphGather(batch_size, activation="tanh"))

if regress == 1:
    model = mymodel2.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_features,
        logdir=logdir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        learning_rate_decay_time=decay_step,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)
else:
    model = mymodel.MultitaskGraphClassifier(
        graph_model,
        len(tasks),
        n_features,
        final_loss='focal_loss',
        yita=args.yita,
        logdir=logdir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        learning_rate_decay_time=decay_step,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

# Fit trained model
# hiv_datasets: 41913;
# train_data: 33536;
# steps_every_epoch:262;
tf.reset_default_graph()
model.fit(train_dataset,
          valid_dataset,
          test_dataset,
          metric,
          transformers,
          nb_epoch=args.nb_epoch,
          max_checkpoints_to_keep=5,
          log_every_N_batches=train_step,
          checkpoint_interval=train_step)
rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean)
print("Evaluating model")
if regress == 1:
    train_scores = model.evaluate(train_dataset, [metric,rms,mae], transformers)
else:
    train_scores = model.evaluate(train_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

f = open(logdir + '.log','a')
f.write(str(train_scores))
f.close()   
