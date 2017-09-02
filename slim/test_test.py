# -*- coding: utf-8 -*-
from test import testResult

if __name__ == '__main__':
    #test
    print(testResult(checkpoint_path="/tmp/model/",
                     model_name="mobilenet_v1",
                     num_classes=5,
                     img_path="/img/testdata/business/533243822.jpeg"))
