
#### Data Resource
The dataset can be downloaded from [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

If you would like to download and process the audio dataset using python script, please fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSdQhpq2Be2CktaPhuadUMU7ZDJoQuRlFlzNO45xO-drWQ0AXA/viewform?fbzx=7440236747203254000) to request a user name and password, then pass them as arguments to examples/speaker/voxceleb/local/prepare_data.py.

#### Comparasion with Published Results
Model                |   Top1-Acc (%)|   Top5-Acc (%) |    Dimensions  | Aggregation |   Parameters  |
|:-                  |:-:            |:-:             |:-:             |:-:          |:-:            |
VGG-M [[1]](#1)      |   80.5        |   92.1         |    1024        |   TAP       |   67 million  |
ResNet-34 [[2]](#2)  |   88.5        |   94.9         |    128         |   TAP       |   1.4 million |
ResNet-34 [[2]](#2)  |   89.2        |   95.1         |    128         |   SAP       |   1.4 million |
ResNet-34 [[2]](#2)  |   89.9        |   95.7         |    128         |   LDE       |   1.4 million |


#### References
<a id="1">[1]</a> VoxCeleb: a large-scale speaker identification dataset
<a id="2">[2]</a> Exploring the Encoding Layer and Loss function in End-to-End
