# -*- coding: utf-8 -*-
from train import createModel,stop,restart,relearn
from convert import createTfrecord
from test import testResult
from visualize import createVisualization

if __name__ == '__main__':
    #relearn model
    relearn(model_id="1",
           train_dir="/tmp/model/",
           dataset_name="labellio",
           dataset_dir="/img/tfrecord/",
           model_name="mobilenet_v1",
           max_number_of_steps=1000,
           batch_size=10,
           learning_rate=0.95,
           learning_rate_decay_type="exponential",
           optimizer="rmsprop",
           model_every_n_steps=1000,
           learning_rate_decay_factor=0.90,
           decay_steps=10,
           utilization_per_gpu=0.95,
           gpu_number="0")
