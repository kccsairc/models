# -*- coding: utf-8 -*-
from train import createModel,stop,restart,relearn

if __name__ == '__main__':
    #create model
    createModel(train_dir="/tmp/model/",
                dataset_name="labellio",
                dataset_dir="/img/tfrecord/",
                num_train=10000,
                num_val=1000,
                num_classes=5,
                model_name="mobilenet_v1",
                max_number_of_steps=100,
                batch_size=10,
                learning_rate=0.01,
                learning_rate_decay_type="exponential",
                optimizer="rmsprop",
                model_every_n_steps=10,
                learning_rate_decay_factor=0.90,
                decay_steps=10,
                utilization_per_gpu=1.0,
                gpu_number="0",
                checkpoint_path="/model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt")
