# -*- coding: utf-8 -*-
from visualize import createVisualization

if __name__ == '__main__':
    #deep visualization
    print(createVisualization(checkpoint_path="/tmp/model/",
                              model_name="mobilenet_v1",
                              num_classes=2,
                              img_path="/img/testdata2/dog/1.jpg"))
