# -*- coding: utf-8 -*-
from train import createModel,stop,restart,relearn

if __name__ == '__main__':
    #stop model
    stop(train_dir="/tmp/model/")
    #stop model using status manager
    #stop(model_id="1")
