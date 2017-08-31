# -*- coding: utf-8 -*-
from visualize import createVisualization

if __name__ == '__main__':
    #deep visualization
    print(createVisualization(checkpoint_path="/tmp/model/model__step30.ckpt",model_name="mobilenet_v1",num_classes=5,img_path="/img/testdata/old_clothes/1456541309.jpeg"))
