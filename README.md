# MAML few-shot classification

This repository is containing the code to complete a project on few-shot remote sensing image classification.
To this extent a simple CNN network, as shown in the following image, is used with the [model-agnostic meta-learning approach](https://arxiv.org/abs/1703.03400) with multiple update steps.
![model](/images/model.png)
A small MAML recap, based on the origianl paper, is presented in the next figure. ![maml](/images/maml.png)

# Datasets

To investigate this, two dataset were used. The [Floating Object Dataset](https://github.com/ESA-PhiLab/floatingobjects/tree/master) from _ESA-PhiLab_ and the [2020 Data Fusion Contest Dataset](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest). 

## Floating Object Dataset

To create the few-shot dataset setup some pre-processing had to be done, specially for the floatingobject dataset. Indeed, as this dataset is used for object segmentation, artificial classes had to be created. 

<br> For this dataset the classes are created as regions from a given area as shown in the following figure where there is four contrastive classes and 10 shots. This is then organised in multiple _tasks_ each having different artificial classes. Those classes are then used during meta-training to achieve the few-shot adaptation. ![floatingobjectclasses](/images/overview.png)

Then for testing the model is adapted on the few-shot images before being asked to classify the new images. However, as the classes are very artificial, the results are not very satisfactory. Then the model is very dependant to the task, i.e. the artifical classes chosen, as the classes are encoding very diverse informations.
![hard FO](/images/hard_FO.png)
The results on three regions are shown bellow where _South Africa_ is only used at testing where the two others are used during training.
<table>
  <tr>
    <td> <b>Region</td>
    <td> <b>1-Shot</td>
    <td> <b>5-Shots</td>
    <td> <b>10-Shots</td>
  </tr>
  <tr>
    <td> South Africa </td>
    <td> 25.6 </td>
    <td> 29.2 </td>
    <td> 31.6 </td>
  </tr>
  <tr>
    <td> Mandaluyong </td>
    <td> 44.8 </td>
    <td> 45.2 </td>
    <td> 52.0 </td>
  </tr>
  <tr>
    <td> Vungtau </td>
    <td> 22.0 </td>
    <td> 21.6 </td>
    <td> 24.4</td>
  </tr>
</table>

## Data Fusion Contest 2020 Dataset

The Data Fusion Contest Dataset is made for multi-class remote sensing image segmentation. However, as this project aim is to investigate MAML few-shot classification, a self-supervised training approach is chosen. Hence, during training contrastive classes are used and the real classes are adapted to and used during testing where is image class is the most present one on the image.
<br>
As for the floating object dataset, the contrastive classes are created based on geographical localisation. As the dataset is already separated in tiles, those will be the "shots" of each contrastive class. To limit the inter-class similarity, a safety maring is added to have sufficient distances between the contrastive classes. To increase the number tasks, each region is used multiple time to randomly select constrastive classes on each region.

<br> To further reduce the contrastive class similarity issue, a similarity based contrastive class creation process has been tested. Indeed when plotting the intra-class to inter-class cosine similartiy ratio, next figure, an important number of classes have a ratio smaller than one. Hence during training wiht the improved process, those classes with ratio lower than one are skipped. In addition, the outer-loss is weighted using the intra-class similartiy to improve training.
![similarity ratio](/images/ratio.png)

For comparison purposes, a simple ResNet model trained using the labels is used. For the self-supervised models, the testing is using the labeled images with multiple gradient steps to adapt to the various classes.
<table>
  <tr>
    <td> <b>Training</td>
    <td> <b>1-Shot</td>
    <td> <b>5-Shots</td>
    <td> <b>10-Shots</td>
  </tr>
  <tr>
    <td> Labels </td>
    <td> 59.04 </td>
    <td> 77.54 </td>
    <td> 78.74 </td>
  </tr>
  <tr>
    <td> Self-Supervised </td>
    <td> 43.60 </td>
    <td> 61.43 </td>
    <td> 67.04 </td>
  </tr>
  <tr>
    <td> Self-Supervised <br> Improved </td>
    <td> 51.00 </td>
    <td> 63.90 </td>
    <td> 67.95</td>
  </tr>
</table>

# Note
The data are assumed to be located in: DATA\DFC\DFC_Public_Dataset\ROIs0000_... folders or DATA\FO_DATA\base_image
