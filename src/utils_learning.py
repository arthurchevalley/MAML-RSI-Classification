
import os
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix

import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import rasterio

l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Dataset regions definition

regions_FO = ["accra_20181031",
               "biscay_20180419",
               "danang_20181005",
               "lagos_20200505",
               "mandaluyong_20180314",
               "neworleans_20200202",
               "panama_20190425",
               "portalfredSouthAfrica_20180601",
               "sanfrancisco_20190219",
               "shengsi_20190615",
               "tangshan_20180130",
               "toledo_20191221",
               "turkmenistan_20181030",
               "venice_20180630",
               "venice_20180928",
               "vungtau_20180423"
               ]

regions_DFC = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]

# DFC dataset classes definition
all_classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water", "other"])

IGBP_simplified_class_mapping = [
    0,  # Evergreen Needleleaf Forests
    0,  # Evergreen Broadleaf Forests
    0,  # Deciduous Needleleaf Forests
    0,  # Deciduous Broadleaf Forests
    0,  # Mixed Forests
    1,  # Closed (Dense) Shrublands
    1,  # Open (Sparse) Shrublands
    2,  # Woody Savannas
    2,  # Savannas
    3,  # Grasslands
    4,  # Permanent Wetlands
    5,  # Croplands
    6,  # Urban and Built-Up Lands
    5,  # Cropland Natural Vegetation Mosaics
    7,  # Permanent Snow and Ice
    8,  # Barren
    9,  # Water Bodies
    10
]



def create_CM(args, trg_cm, predi_cm, current_class, ds, number_all_classes_training):
    """
    Create the testing confusion matrix.
    :param args: arg parser provided by the user
    :param trg_cm: Target classes
    :param predi_cm: Predicted classes
    :param current_class: Class if one versus all approach
    :param ds: Dataset class
    :param number_all_classes_training: Number of classes at training
    :return: Confusion matrix
    """
    cf_matrix = confusion_matrix(trg_cm, predi_cm)

    ratio = ds.ratio_all_classes.copy()[current_class]/np.sum(ds.ratio_all_classes.copy())
    ratio_adapt = number_all_classes_training[0, current_class]/number_all_classes_training[1,current_class]
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 1, index=[i for i in range(len(set(trg_cm)))],
                        columns=[i for i in range(len(set(trg_cm)))])
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted class")
    plt.ylabel("Ground truth")
    image_name = os.path.join(args.output_folder,
                              'OVA_ConfusionMatrix_{0}Shots_{1}Ways_Class{2}_GS{3}.png'.format(
                                  args.num_shots,
                                  args.num_ways,
                                  current_class,
                                  args.gradient_steps
                              ))
    plt.savefig(image_name)
    plt.show()


def split_support_query(ds, shots, random_state=0, at_least_n_queries=0, FO = False, nbr = None):
    """
    Retrieve the number of images wanted to match the number of shots
    :param ds: Dataset
    :param shots: Number of shots wanted
    :param random_state: the random seed
    :param at_least_n_queries: The minimum number of images required
    :param FO: Use of the Floating object dataset or not
    :param nbr: Number of test images wanted
    :return: Adaptation training and testing images
    """
    classes, counts = np.unique(ds.index.maxclass, return_counts=True)
    # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset
    classes = classes[counts > (shots + at_least_n_queries)]
    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.maxclass == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)

    supports = pd.concat(supports)
    if FO:
        support_data = [ds.index.loc[ds.index["index"] == idx]["files"][0] for idx in supports["index"].to_list()]
        support_data_class = [ds.index.loc[ds.index["index"] == idx]["maxclass"][0] for idx in supports["index"].to_list()]

        support_input = []
        support_target = []
        for i,f in enumerate(support_data):
            with rasterio.open(os.path.join(ds.root, f), "r") as src:

                support_input.append(src.read())
            support_target.append(support_data_class[i])
        support_input = np.stack(support_input)
        if support_input.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            support_input = support_input[:, idx]
        support_input = torch.from_numpy(support_input.astype(np.int32))
        support_target = np.stack(support_target)
        support_target = torch.from_numpy(support_target.astype(np.int32))
    else:
        support_data = [ds[idx] for idx in supports["index"].to_list()]

        support_input, _ = list(zip(*support_data))
        support_dfc_labels = supports.maxclass.values

        support_input = torch.stack(support_input)
        support_target = torch.from_numpy(support_dfc_labels)

    # query
    queries = pd.concat(queries)
    if FO:
        if queries.shape[0] > (nbr*6):
            idx_queries = queries["index"].to_list()
            idx_queries = random.sample(idx_queries, (nbr*6))
        else:
            idx_queries = queries["index"].to_list()
        query_data = [ds.index.loc[ds.index["index"] == idx]["files"][0] for idx in idx_queries]
        query_data_class = [ds.index.loc[ds.index["index"] == idx]["maxclass"][0] for idx in idx_queries]

        query_input = []
        query_target = []
        for i, f in enumerate(query_data):
            with rasterio.open(os.path.join(ds.root, f), "r") as src:
                query_input.append(src.read())
            query_target.append(query_data_class[i])
        query_input = np.stack(query_input)
        if query_input.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            query_input = query_input[:, idx]
        query_target = np.stack(query_target)
        query_input = torch.from_numpy(query_input.astype(np.int32))
        query_target = torch.from_numpy(query_target.astype(np.int32))
        return support_input, support_target, query_input, query_target
    else:
        query_data = [ds[idx] for idx in queries["index"].to_list()]

        query_input, _ = list(zip(*query_data))
        query_input = torch.stack(query_input)
        query_target = torch.from_numpy(queries.maxclass.values)
        a = queries.multi_class.values.tolist()
        query_target_all = []
        for i in range(len(a)):
            b = a[i]
            skip = False
            d = []
            for j in range(len(b)):
                if j == (len(b)- 1) or j == 0 or skip:
                    skip = False
                    continue
                if (b[j] != ' ' and b[j] != ']') and (b[j+1] != ' ' and b[j+1] != ']'):
                    d.append(int(b[j:j+2]))
                    skip = True
                elif b[j] != ' ':
                    d.append(int(b[j]))
                else:
                    continue
            query_target_all.append(d)

        return support_input, support_target, query_input, query_target,query_target_all



def conv3x3(in_channels, out_channels, **kwargs):
    """
    Convolutional block of the model
    :param in_channels: Number of channel at input
    :param out_channels: Number of output channel
    :param kwargs: Other param possible
    :return: A conv block composed of conv2D, Batch norm, ReLU and MaxPool
    """
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        #features = features.view((features.size(0), -1))
        features = features.mean(-1).mean(-1)
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def get_hist(input, second_input = None):
    """
    Compute the histogram of a given series of array
    :param input: Input arrays to be transformed in a histogram
    :param second_input: If two set of images are concatenated in a single histogram, this is the second images serie
    :return: Histograms of the given images
    """
    in_shape = input.shape[0]
    flatten_input_single = torch.flatten(input, start_dim=2)
    if second_input is not None:
        flatten_input_second = torch.flatten(second_input, start_dim=2)
        flatten_input_single = torch.cat([flatten_input_second, flatten_input_single], dim=0)

    mean = torch.mean(flatten_input_single, dim=1)
    std = torch.std(flatten_input_single, dim=1)
    historgram = torch.stack([mean, std])
    return historgram


def get_accuracy_OVA(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points for the one versus all case
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    probas = torch.sigmoid(logits)
    predictions = probas.argmax(0)
    return torch.mean(predictions.eq(targets).float())


def get_accuracy_BIN(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    predictions = torch.round(torch.sigmoid(logits))
    return torch.mean(predictions.eq(targets).float())


class Similartiy_class():
    def __init__(self, input_shape, sample_number,temperature=0.1):
        super(Similartiy_class, self).__init__()
        self.temperature = temperature
        self.cos_sim_score = torch.zeros(input_shape,sample_number)
        self.sim_metric = torch.zeros(input_shape)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.samples = None
        self.input = None

    def compute_loss(self, input, positive, negative):
        """
        Compute a NCE-like loss for similarity measurement of two flatten images.
        ! DO NOT DO THAT as there is the spatial bias
        :param input: Ensemble of the samples to compare
        :param positive: Positive samples
        :param negative: Negative samples
        :return: NCE-like metric
        """
        neg_flat = torch.flatten(negative, start_dim=1)
        pos_flat = torch.flatten(positive, start_dim=0)
        pos_flat = torch.unsqueeze(pos_flat, 0)
        self.samples = torch.cat([pos_flat, neg_flat], dim=0)
        self.input = torch.flatten(input, start_dim=1)
        self.cos_sim_score = torch.zeros(input.shape[0], pos_flat.shape[0] + neg_flat.shape[0])
        self.nce_loss = torch.zeros(input.shape[0])
        for i in range(input.shape[0]):
            aa = torch.unsqueeze(self.input[i,:], 0)
            cc = self.cos_sim(aa, self.samples.float())
            self.cos_sim_score[i,:] = cc
            pos = np.exp(self.cos_sim_score[i,0]/self.temperature)
            neg = np.exp(self.cos_sim_score[i,1:]/self.temperature).sum()
            self.nce_loss[i] = pos/(pos+neg)#-np.log(pos/(pos+neg))
        return self.nce_loss

    def compute_loss_hist_pos(self, input):
        """
        Compute the intra-class similarity of two histograms
        :param input: Positive samples
        :param neg_idx: Negative samples indexes
        :return: The mean intra-class similartiy for the batch of histogram given
        """
        sim_metric = torch.zeros(input.shape[1])
        cos_sim = torch.nn.CosineSimilarity(dim=1)

        for i in range(input.shape[1]):
            if i != input.shape[1] - 1:
                pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0),
                                            torch.cat([input[0][:i], input[0][(i + 1):]])))
                pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0),
                                           torch.cat([input[1][:i], input[1][(i + 1):]])))
            else:
                if i == 0:
                    pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0), torch.unsqueeze(input[0,i,:], dim=0)))
                    pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0), torch.unsqueeze(input[1,i,:], dim=0)))

                else:
                    pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0),input[0][:i]))
                    pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0),input[1][:i]))

            pos_cos_sim = (pos_cos_sim_mean + pos_cos_sim_std)/2
            sim_metric[i] = pos_cos_sim.mean()
        sim_metric = sim_metric.mean()
        return sim_metric

    def compute_loss_hist_negativ(self, input, anchor, neg_idx):
        """
        Compute the inter-class similarity of histograms
        :param input: Positive samples' histogram
        :param anchor: Negative samples histograms
        :param neg_idx: indexes of the negatives histograms
        :return: The inter-class similarity
        """
        neg_hist = anchor[:, 0:len(neg_idx), :]
        sim_metric = torch.zeros(input.shape[1])
        cos_sim = torch.nn.CosineSimilarity(dim=1)

        for i in range(input.shape[1]):
            neg_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0), neg_hist[0]))
            neg_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0), neg_hist[1]))
            neg_cos_sim = (neg_cos_sim_mean + neg_cos_sim_std) / 2
            loss = neg_cos_sim
            sim_metric[i] = loss.mean()

        sim_metric = sim_metric.mean()
        return sim_metric

    def compute_loss_hist(self, input, anchor, neg_idx):
        """
        NCE loss as similarity measure for positives, i.e. pos/(pos+neg)
        :param input: Positive class samples' histograms
        :param anchor: All samples
        :param neg_idx: Negatives' class indexes
        :return: mean nce-like loss
        """
        neg_hist = anchor[:, 0:len(neg_idx), :]
        pos_hist = anchor[:, len(neg_idx):, :]
        nce_loss = torch.zeros(input.shape[1])
        cos_sim = torch.nn.CosineSimilarity(dim=1)

        for i in range(input.shape[1]):
            if i != input.shape[1] - 1:
                pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0),
                                            torch.cat([input[0][:i], input[0][(i + 1):]])))
                pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0),
                                           torch.cat([input[1][:i], input[1][(i + 1):]])))
            else:
                if i == 0:
                    pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0), torch.unsqueeze(input[0,i,:], dim=0)))
                    pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0), torch.unsqueeze(input[1,i,:], dim=0)))

                else:
                    pos_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0, i, :], dim=0),input[0][:i]))
                    pos_cos_sim_std = (cos_sim(torch.unsqueeze(input[1, i, :], dim=0),input[1][:i]))

            neg_cos_sim_mean = (cos_sim(torch.unsqueeze(input[0,i,:],dim=0), neg_hist[0]))
            neg_cos_sim_std = (cos_sim(torch.unsqueeze(input[1,i,:],dim=0), neg_hist[1]))

            pos_cos_sim = (pos_cos_sim_mean + pos_cos_sim_std)/2
            neg_cos_sim = (neg_cos_sim_mean + neg_cos_sim_std) / 2

            pos = np.exp(pos_cos_sim.cpu()/self.temperature).mean()
            neg = np.exp(neg_cos_sim.cpu()/self.temperature).mean()
            loss = pos/neg
            nce_loss[i] = loss

        nce_loss = nce_loss.mean()
        return nce_loss

    def compute_loss_hist_neg(self, input, anchor, neg_idx):
        """
        NCE loss as similarity measure for positives, i.e. neg/(pos+neg)
        :param input: Positive class samples' histograms
        :param anchor: All samples
        :param neg_idx: Negatives' class indexes
        :return: mean nce-like loss
        """
        neg_hist = anchor[:,0:len(neg_idx),:]
        pos_hist = anchor[:,len(neg_idx):,:]
        nce_loss = torch.zeros(anchor.shape[1])
        cos_sim = torch.nn.CosineSimilarity(dim=1)

        for i in range(anchor.shape[1]):
            if i != anchor.shape[1]-1:
                pos_cos_sim_mean = (cos_sim(torch.unsqueeze(anchor[0,i,:],dim=0), torch.cat([anchor[0][:i],anchor[0][(i+1):]])))
                pos_cos_sim_std = (cos_sim(torch.unsqueeze(anchor[1,i,:],dim=0), torch.cat([anchor[1][:i],anchor[1][(i+1):]])))

            else:
                pos_cos_sim_mean = (cos_sim(torch.unsqueeze(anchor[0, i, :], dim=0),torch.unsqueeze(anchor[0][:i], dim=0)))
                pos_cos_sim_std = (cos_sim(torch.unsqueeze(anchor[1, i, :], dim=0),torch.unsqueeze(anchor[1][:i], dim=0)))
            if input.shape[1] != 0:
                neg_cos_sim_mean = (cos_sim(torch.unsqueeze(anchor[0,i,:],dim=0), input[0]))
                neg_cos_sim_std = (cos_sim(torch.unsqueeze(anchor[1,i,:],dim=0), input[1]))
            else:
                neg_cos_sim_mean = torch.zeros_like(pos_cos_sim_mean)
                neg_cos_sim_std = torch.zeros_like(pos_cos_sim_std)

            pos_cos_sim = (pos_cos_sim_mean + pos_cos_sim_std)/2
            neg_cos_sim = (neg_cos_sim_mean + neg_cos_sim_std) / 2
            pos = np.exp(pos_cos_sim.cpu()/self.temperature).mean()
            neg = np.exp(neg_cos_sim.cpu()/self.temperature).mean()
            loss = pos/(pos+neg)
            nce_loss[i] = loss

        nce_loss = nce_loss.mean()
        return nce_loss
