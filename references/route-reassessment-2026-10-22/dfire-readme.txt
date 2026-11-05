## :fire: D-Fire: an image dataset for fire and smoke detection

**Authors:** Researchers from Gaia, solutions on demand ([GAIA](https://www.gaiasd.com/))

### :bulb: About

D-Fire is an image dataset of fire and smoke occurrences designed for machine learning and object detection algorithms. It contains more than 21,000 images, compiled from our own surveillance cameras, the web, and public datasets. A summary of its composition is shown below.

<div align="center">
<table>
  <tr>
    <th>Number of images</th>
    <th>Number of bounding boxes</th>
  </tr>
 
  <tr><td>

  | Category | # Images |
  | ------------- | ------------- |
  | Only fire  | 1,164  |
  | Only smoke  | 5,867  |
  | Fire and smoke  | 4,658  |
  | None  | 9,838  |

  </td><td>

  | Class | # Bounding boxes |
  | ------------- | ------------- |
  | Fire  | 14,692 |
  | Smoke  | 11,865 |

  </td></tr> 
</table>
</div>

Annotations follow the YOLO format, with normalized coordinates between 0 and 1.
To facilitate usage, we provide a `utils.yolo2pixel` function to convert these normalized coordinates into pixel coordinates.

***

### :high_brightness: Example images

Below are some representative samples from the D-Fire dataset showcasing different scenarios of fire and smoke detection. 

<p align="center">
    <img width="500" src="./figures/dfire_examples.png" alt="D-Fire examples">
<p>

### :computer: Download links

Access the D-Fire dataset, including images, annotations, and pre-split training, validation, and test sets via the following links. Additional resources such as surveillance videos and trained models are also available to support your research.

* [D-Fire dataset (only images and labels)](https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw).
* [Training, validation and test sets](https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN).
* [Kaggle version ready to use](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo).
* [Some surveillance videos](https://1drv.ms/f/c/c0bd25b6b048b01d/EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ). 
* [Some deep learning models trained with the D-Fire dataset](https://github.com/pedbrgs/Fire-Detection).
* For more surveillance videos, register on our platform [Apaga o Fogo (AoF)](https://apagaofogo.eco.br/).

### :globe_with_meridians: Public dataset images

Images whose filenames start with the prefix `PublicDataset` were compiled from third-party public sources, including:

* [HPWREN](https://hpwren.ucsd.edu/) (High Performance Wireless Research and Education Network, UC San Diego): frames from wildfire camera time-lapses. See also the [Fire Ignition images Library (FIgLib)](https://legacy-www.hpwren.ucsd.edu/FIgLib/).
* [Wildfire Observers and Smoke Recognition](http://wildfire.fesb.hr/) project, Center for Wildfire Research, FESB, University of Split, Croatia: images from its [image database](http://wildfire.fesb.hr/index.php?option=com_content&view=article&id=49&Itemid=54) and [video database](http://wildfire.fesb.hr/index.php?option=com_content&view=article&id=50&Itemid=55).
* [FireNet](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) dataset ([Jadon et al., 2019](https://arxiv.org/abs/1905.11922)): fire and non-fire images from its training set.

These images remain subject to the terms of their original providers. Users, especially for commercial use, should check those terms and credit the original sources. In particular, FESB asks users to credit the Center for Wildfire Research.

### :scroll: Citation

Please cite the following paper if you use our image data collection:

- <p align="justify">Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa: <a href="https://link.springer.com/article/10.1007/s00521-022-07467-z"> An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. </a> In: Neural Computing and Applications, 2022.</p>

If you use our surveillance videos, please cite the following paper:
- <p align="justify"><b>Pedro Vinícius Almeida Borges de Venâncio</b>, Roger Júnio Campos, Tamires Martins Rezende, Adriano Chaves Lisboa, Adriano Vilela Barbosa: <a href="https://link.springer.com/article/10.1007/s00521-023-08260-2"> A hybrid method for fire detection based on spatial and temporal patterns. </a> In: Neural Computing and Applications, 2023.</p>
