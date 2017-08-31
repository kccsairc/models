# -*- coding: utf-8 -*-
from train import restart

if __name__ == '__main__':
    #create model
    restart(train_dir="/tmp/model/",
            dataset_name="labellio",
            dataset_dir="/img/tfrecord/",
            model_name="mobilenet_v1",
            max_number_of_steps=1000,
            batch_size=10,
            learning_rate=0.01,
            learning_rate_decay_type="fixed",
            optimizer="rmsprop",
            model_every_n_steps=10,
            decay_steps=10,
            utilization_per_gpu=1.0,
            gpu_number="0",
            checkpoint_path="/model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt")
