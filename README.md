# Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks
This repository contains the source code for the paper [Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks](https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks).

## Abstract
In DCNNs, the number of parameters in pointwise convolutions rapidly grows due to the multiplication of the number of filters by the number of input channels that come from the previous layer. Our proposal makes pointwise convolutions parameter efficient via grouping filters into parallel branches or groups, where each branch processes a fraction of the input channels. However, by doing so, the learning capability of the DCNN is degraded. To avoid this effect, we suggest interleaving the output of filters from different branches at intermediate layers of consecutive pointwise convolutions. We applied our improvement to the EfficientNet, DenseNet-BC L100, MobileNet and MobileNet V3 Large architectures. We trained these architectures with the CIFAR-10, CIFAR-100, Cropped-PlantDoc and The Oxford-IIIT Pet datasets. When training from scratch, we obtained similar test accuracies to the original EfficientNet and MobileNet V3 Large architectures while saving up to 90% of the parameters and 63% of the flops.

## Citing this Paper 
```
@article{Schwarz Schuler_Romani_Abdel-Nasser_Rashwan_Puig_2022, 
  title={Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks}, volume={28}, 
  url={https://mendel-journal.org/index.php/mendel/article/view/169}, 
  number={1}, 
  journal={MENDEL}, 
  author={Schwarz Schuler, Joao Paulo and Romani, Santiago and Abdel-Nasser, Mohamed and Rashwan, Hatem and Puig, Domenec},
  year={2022}, month={Jun.}, pages={23-31} }
```

## Test on Colab
You can test kEffNet V1 via Google Colab:
* [kEffNet v1.](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/kEffNet_v1.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/kEffNet_v1.ipynb)
