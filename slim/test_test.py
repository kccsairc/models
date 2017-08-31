# -*- coding: utf-8 -*-
from test import testResult

if __name__ == '__main__':
    #test
    print(testResult(checkpoint_path="/tmp/tfmodel/model.ckpt",model_name="mobilenet_v1",num_classes=5,img_path="/img/testdata/old_clothes/1456541309.jpeg"))
