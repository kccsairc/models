# -*- coding: utf-8 -*-
from train import createModel,stop,restart,relearn

if __name__ == '__main__':
    #create model
    createModel(model_id=1,
                train_dir="/tmp/model/",
                dataset_name="labellio",
                dataset_dir="/img/tfrecord/",
                num_train=200,
                num_val=40,
                num_classes=5,
                model_name="mobilenet_v1",
                max_number_of_steps=1000,
                batch_size=10,
                learning_rate=0.01,
                learning_rate_decay_type="exponential",
                optimizer="rmsprop",
                model_every_n_steps=1000,
                learning_rate_decay_factor=0.90,
                decay_steps=10,
                utilization_per_gpu=0.95,
                gpu_number="0",
                checkpoint_path="/model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt")
