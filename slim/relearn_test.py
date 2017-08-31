# -*- coding: utf-8 -*-
from train import createModel,stop,restart,relearn
from convert import createTfrecord
from test import testResult
from visualize import createVisualization

if __name__ == '__main__':
    #create model
    #createModel(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir="/img/tfrecord/",model_name="mobilenet_v1",max_number_of_steps=1000,batch_size=10,learning_rate=0.01,learning_rate_decay_type="fixed",optimizer="rmsprop",model_every_n_steps=10,weight_decay=0.00004,decay_steps=10,utilization_per_gpu=1.0,gpu_number="0",checkpoint_path=None)
    #stop model

    #restart model

    #relearn model
    createModel(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir="/img/tfrecord/",model_name="mobilenet_v1",max_number_of_steps=200,batch_size=10,learning_rate=0.01,learning_rate_decay_type="exponential",optimizer="rmsprop",model_every_n_steps=10,learning_rate_decay_factor=0.90,decay_steps=10,utilization_per_gpu=1.0,gpu_number="0")

    #create tfrecord
    #createTfrecord(input_dir="/img/testdata/", tfrecord_dir="/img/tfrecord/", num_data=5,validations=30,dataset_name="labellio")

    #test
    #print(testResult(checkpoint_path="/tmp/tfmodel/model.ckpt",model_name="mobilenet_v1",num_classes=5,img_path="/img/testdata/old_clothes/1456541309.jpeg"))

    #deep visualization
    #print(createVisualization(checkpoint_path="/tmp/tfmodel/model.ckpt",model_name="mobilenet_v1",num_classes=5,img_path="/img/testdata/old_clothes/1456541309.jpeg"))
