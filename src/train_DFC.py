import csv

from utils.dataset_utils import DFCDataset, DFCDataset_contrastive
from torch.utils.data import DataLoader
from utils.ResNet_model import ResNet
import os
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import logging
from collections import OrderedDict

from sklearn.metrics import confusion_matrix


import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from utils_learning import *

logger = logging.getLogger(__name__)

l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Dataset regions definition

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

def get_classification_transform(s2only):
    """
    Data preparation and augmentation for the DFC dataset
    :param s2only: Bool to determine wether to use only the s2 band or not
    :return: prepared data
    """
    def transform(s1, s2, label):
        s2 = s2 * 1e-4

        if s2only:
            input = s2
        else:
            s1 = s1 * 1e-2
            input = np.vstack([s1, s2])

        igbp_label = np.bincount(label.reshape(-1)).argmax() - 1
        target = IGBP_simplified_class_mapping[igbp_label]

        if np.random.rand() < 0.5:
            input = input[:, ::-1, :]

        # horizontal flip
        if np.random.rand() < 0.5:
            input = input[:, :, ::-1]

        # rotate
        n_rotations = np.random.choice([0, 1, 2, 3])
        input = np.rot90(input, k=n_rotations, axes=(1, 2)).copy()

        if np.isnan(input).any():
            input = np.nan_to_num(input)
        assert not np.isnan(target).any()

        return torch.from_numpy(input), target

    return transform


def DFC_tasks(args,region_id,nbr_tasks,SF_range, all = False):
    """
    Create the contrastive DFC tasks
    :param args: arg parser provided by the user
    :param region_id: Region on which the tasks are created
    :param nbr_tasks: Number of tasks
    :param SF_range: Safety margin around a given class
    :param all: Bool to indicate if all region are used to create tasks
    """

    path = os.path.dirname(os.path.dirname(__file__)) + '/DATA/DFC/DFC_Public_Dataset/'
    transform = get_classification_transform(s2only=True)
    ds = DFCDataset_contrastive(path, region=regions_DFC[region_id], transform=transform,num_ways=args.num_cways)

    # Tasks list file name
    if all:
        file_name = os.path.join(path+'{:}tasks_{:}cways_NCE_all_{:}MT_all_bigSF.csv'.format(nbr_tasks, args.num_cways, ds.multitask))
    else:
        file_name = os.path.join(path+'{:}tasks_{:}cways_NCE_all_{:}MT_region{:}.csv'.format(nbr_tasks,args.num_cways,ds.multitask,region_id))

    # Tasks creation and save in the index file
    for task in tqdm(range(nbr_tasks)):
        classes_id = ds.contrastive_classes(regions_DFC[region_id],SF_range)
        with open(file_name, 'a', newline='') as f:
            write = csv.writer(f)
            write.writerows(classes_id)


def read_DFC_cont(num_ways = 4, file_name='Initial'):
    """
    Read the DFC contrastive tasks index file
    :param num_ways: Number of contrastive tasks
    :param file_name: File name containing DFC contrastive tasks
    :return: Classes indexes
    """
    
    path = os.path.dirname(os.path.dirname(__file__)) + '/DATA/DFC/DFC_Public_Dataset/'
    if file_name == 'Initial':
        file_name = os.path.join(path+'{:}tasks_{:}cways_NCE_all_{:}MT_all.csv'.format(20, 4, 1))
    elif file_name == 'Big_SF_bigger':
        file_name = os.path.join(path+'{:}tasks_{:}cways_NCE_all_{:}MT_all_shuffle_bigSF.csv'.format(30, 4, 1))
    elif file_name == 'Big_SF_Biggest':
        file_name = os.path.join(path+'{:}tasks_{:}cways_NCE_all_{:}MT_all_bigSF.csv'.format(40, 4, 1))
    else:
        file_name = file_name

    # Define the number of classes. Must match the numer defined when creating the file
    nbr_classes = num_ways
    with open(file_name, 'r',newline='') as f:
        csv_reader = csv.reader(f)
        classes = -np.ones((nbr_classes, 1), dtype = object)
        classes_tmp = -np.ones((nbr_classes, 1), dtype=object)
        for line,row in enumerate(csv_reader):
            row_item = []
            for i in range(len(row)):
                if int(row[i]) != -1:
                    row_item.append(int(row[i]))
            classes_tmp[line%nbr_classes, 0] = row_item

            if line%nbr_classes == (nbr_classes - 1):
                if line == (nbr_classes - 1):
                    classes[:, 0] = classes_tmp[:,0].copy()
                elif line == (2*nbr_classes - 1):
                    classes = np.stack([classes, classes_tmp])
                else:
                    classes = np.concatenate([classes,np.expand_dims(classes_tmp, axis=0)])
                classes_tmp = -np.ones((nbr_classes, 1), dtype=object)
    return classes[:, :, 0]


def train_oneversusall_DFC_pred(args, learning_rate, step_size):
    """
    Train a model on DFC dataset using one versus all approach
    :param args: arg parser provided by the user
    :param learning_rate: Training learning rate
    :param step_size: Training step size
    """

    path = os.path.dirname(os.path.dirname(__file__)) + '/DATA/DFC/dfc_0/'
    transform = get_classification_transform(s2only=True)

    torch.cuda.empty_cache()

    # Retrieve contrastive classes id
    constrative_ids = read_DFC_cont(args.num_cways, args.DFC_contrastive_tasks)

    hist_file_name = os.path.join(args.output_folder, 'hist_best_hp_DFC.csv')
    ds = DFCDataset_contrastive(path, region=regions_DFC, transform=transform,
                                                        constractive_ids=constrative_ids,
                                                        num_ways=args.num_cways)

    nbr_samples = args.num_shots*args.num_cways
    temperature = 0.5
    loss = Similartiy_class(nbr_samples, args.num_cways, temperature)
    total_ds = len(ds)
    batch_size = args.batch_size
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(13, 1, hidden_size=args.hidden_size)

    model.to(device=args.device)

    model.train()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_print = []
    hist_print = []
    hist_print_pos = []
    hist_print_neg = []

    total_skipped = 0
    total_loader = int(np.floor(total_ds/batch_size))
    if total_loader > args.num_batches:
        total_loader = args.num_batches

    # Training loop
    with tqdm(dataloader, total=total_loader) as pbar:
        for batch_idx, batch in enumerate(pbar):

            torch.cuda.empty_cache()

            model.zero_grad()

            train_inputs, train_targets, test_inputs, test_targets = [t.to(device=args.device) for t in batch]
            train_inputs = train_inputs.float()
            test_inputs = test_inputs.float()
            train_targets = train_targets.float()
            test_targets = test_targets.float()

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            classes = np.unique(train_targets.cpu())
            skip = 0
            # Local parameters update
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                logit = []

                # Random selection of a class to be the positive one for this task
                for c in random.sample(classes.tolist(),1):
                    train_target_OVA = (train_target == c).to(float)
                    test_target_OVA = (test_target == c).to(float)

                # Find positive class corresponding indexes
                idx_pos = list(map(lambda x: x > 0, test_target_OVA))
                pos_idx = []
                neg_idx = []
                for id_add, j in enumerate(idx_pos):
                    if j.item():
                        pos_idx.append(id_add)
                    else:
                        neg_idx.append(id_add)

                # Compute images histograms

                only_pos = True # Either to compute the positive and negative similarity or only intra/inter similarity
                hist_inter = get_hist(test_input[neg_idx, :, :, :])
                hist_intra = get_hist(test_input[pos_idx, :, :, :])

                # Compute the similarities
                if hist_intra.shape[1] != 0 and not only_pos:
                    intra_class_sim = loss.compute_loss_hist(hist_intra, hist_inter, neg_idx)#
                if hist_inter.shape[1] != 0 and not only_pos:
                    intra_class_sim = loss.compute_loss_hist_neg(hist_intra, hist_inter, neg_idx)

                # Intra class similarity
                if only_pos and hist_intra.shape[1] != 0:
                    intra_class_sim = loss.compute_loss_hist_pos(hist_intra)#
                else:
                    intra_class_sim = torch.zeros(1)

                # Inter class similarity
                if only_pos and hist_inter.shape[1] != 0:
                    inter_class_sim = loss.compute_loss_hist_negativ(hist_intra, hist_inter, neg_idx)  #
                else:
                    inter_class_sim = 0.1*torch.ones(1)

                # Compute ratio and save hist
                ratio = intra_class_sim/inter_class_sim
                hist_print.append(ratio.item())
                hist_print_neg.append(inter_class_sim.item())
                hist_print_pos.append(intra_class_sim.item())

                if torch.isnan(ratio) or (ratio.item() < 1.0):
                    skip += 1
                    total_skipped += 1
                    continue

                model.zero_grad()
                meta_optimizer.zero_grad()

                params = OrderedDict()

                for (name, param) in model.named_parameters():
                    params[name] = param

                # Local parameters update
                for t in range(args.gradient_steps):
                    train_logit = model(train_input, params=params)
                    inner_loss = criterion(train_logit.squeeze(1), train_target_OVA)

                    model.zero_grad()
                    grads=torch.autograd.grad(inner_loss, params.values(),
                                              create_graph=not args.first_order)

                    params_next=OrderedDict()
                    for (name, param), grad in zip(list(params.items()), grads):
                        params_next[name] = param - step_size * grad
                    params = params_next

                # Evaluate local parameters on the task test image
                test_logit = model(test_input, params=params)
                test_logit = test_logit.squeeze(1)

                # Outer loss weighting by the intra-class similarity
                outer_loss += F.binary_cross_entropy_with_logits(test_logit, test_target_OVA, weight=intra_class_sim.to(args.device))

                with torch.no_grad():
                    accuracy += get_accuracy_BIN(test_logit, test_target_OVA)

            # Model parameters update. If all tasks have been skipped, no update for this batch
            if skip != args.batch_size:
                outer_loss.div_((args.batch_size-skip))
                accuracy.div_((args.batch_size-skip))
                outer_loss.backward()
                meta_optimizer.step()

                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                accuracy_print.append(accuracy.item())

            else:
                pbar.set_postfix(accuracy='0.0000')

            if batch_idx >= args.num_batches:
                break

        # Number of tasks skipped
        print("total skip {:}".format(total_skipped))
        # Save model
        if args.output_folder is not None:
            model_name = os.path.join(args.output_folder, 'OVA_lr{:}_ss{:}_ratio_histow.th'.format(learning_rate, step_size))
            os.makedirs(os.path.dirname(model_name), exist_ok=True)
            with open(model_name, 'wb') as f:
                state_dict = model.state_dict()
                torch.save(state_dict, f)

            # Save training accuracies
            file_name = os.path.join(args.output_folder, 'Training_curves_DFC_lr{:}_ss{:}.csv'.format(learning_rate, step_size))
            with open(file_name, 'w') as f:
                write = csv.writer(f)
                write.writerow(accuracy_print)
                write.writerow(hist_print_pos)

            # Save histogram similarities
            with open(hist_file_name, 'w') as f:
                write = csv.writer(f)
                write.writerow(hist_print_pos)
                write.writerow(hist_print_neg)
                write.writerow(hist_print)


def test_oneversusall_DFC_pred(args, learning_rate, step_size, region_i):
    """
    Test of the one versus all approach on the DFC dataset
    :param args: arg parser provided by the user
    :param learning_rate: Training learning rate
    :param step_size: Training step size
    :param region_i: Region on which to test
    :return:
    """
    torch.cuda.empty_cache()

    path = os.path.dirname(os.path.dirname(__file__)) + '/DATA/DFC/DFC_Public_Dataset/'
    model_path_folder = os.path.dirname(os.path.dirname(__file__)) + '/models_weights/'


    transform = get_classification_transform(s2only=True)
    ds = DFCDataset(path, region=regions_DFC[region_i], transform=transform)

    batch = len(np.unique(ds.index.maxclass)) * args.num_shots
    dataloader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=args.num_workers)

    if args.model == 'Pretrain':
        k = 1
        model = ResNet(inplanes=13, out_features=1, normtype="instancenorm", avg_pool=True)
        model.to(device=args.device)
        model_file = os.path.join(model_path_folder, 'test_models/maml_s2.pth')
    elif args.model == 'FO_train':
        k = 2
        model = ConvolutionalNeuralNetwork(12,
                                           1,
                                           hidden_size=args.hidden_size)
        model.to(device=args.device)
        # floating train
        model_file = os.path.join(model_path_folder,
                                  'test_models/OVA_resnet_maml_contrastiveMAML_4shot_1way_lr0.01_ss0.285.th')

    elif args.model == 'Best_DFC':

        k = 3
        model = ConvolutionalNeuralNetwork(13,
                                           1,
                                           hidden_size=args.hidden_size)
        model.to(device=args.device)
        # floating train
        model_file = os.path.join(model_path_folder,'test_models/OVA_lr0.005_ss0.25_ratio_histow.th')

    elif args.model == 'Vanilla_DFC':

        k = 4
        model = ConvolutionalNeuralNetwork(13,
                                           1,
                                           hidden_size=args.hidden_size)
        model.to(device=args.device)
        # floating train
        model_file = os.path.join(model_path_folder,'test_models/OVA_lr0.005_ss0.25_vanilla.th')

    else:
        raise ValueError('Model not existing')

    model.load_state_dict(torch.load(model_file, map_location=args.device))
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_print = []
    accuracy_print2 = []

    classes = np.unique(ds.index.maxclass)

    # Retrieve the correct number adaptation and testing images from the dataset
    train_inputs, train_targets, test_inputs, test_targets, _ = split_support_query(ds,args.num_shots)

    # For model trained using FO dataset or ResNet, one must remove B10 band.
    if train_inputs.shape[1] == 13 and (k == 2 or k == 80):
        idx = np.array([l1cbands.index(b) for b in l2abands])
        train_inputs = train_inputs[:, idx]
    if test_inputs.shape[1] == 13 and (k == 2 or k == 80):
        idx = np.array([l1cbands.index(b) for b in l2abands])
        test_inputs = test_inputs[:, idx]

    model.zero_grad()
    train_inputs = train_inputs.float()
    train_targets = train_targets.long()
    nbr_samples = test_targets.shape[0]
    nbr_training_samples = train_targets.shape[0]

    model.train()

    params_all = []

    # Compute a local parameters update for each classes
    for target_class in classes:

        model.zero_grad()

        train_targets_OVA = (train_targets == target_class).to(float)

        params = OrderedDict(model.meta_named_parameters())
        for t in range(args.gradient_steps):
            # Randomly select training input
            idxs = np.random.randint(train_inputs.shape[0], size=int(np.floor(nbr_training_samples)))
            train_logit = model(train_inputs[idxs].float().to(args.device), params=params)

            inner_loss = criterion(train_logit.squeeze(1), train_targets_OVA[idxs].to(args.device))

            model.zero_grad()
            grads = torch.autograd.grad(inner_loss, params.values(),
                                        create_graph=not args.first_order)

            params_next = OrderedDict()
            for (name, param), grad in zip(list(params.items()), grads):
                params_next[name] = param - step_size * grad
            params = params_next
            del train_logit
            torch.cuda.empty_cache()

        params_all.append(params)
    del train_inputs
    torch.cuda.empty_cache()

    # Predictions
    model.eval()
    test_targets_all = []
    pred_label_all = []
    with torch.no_grad():
        # Number of batch at testing
        nbr_tests = 5
        idxs_all = [i for i in range(test_inputs.shape[0])]
        # Number of images per batch at testing
        nbr_to_sample = 50
        for i in range(nbr_tests):
            logits = []

            idxs = list(random.sample(idxs_all, nbr_to_sample))

            test_targets_test = test_targets[idxs].long()
            test_inputs_test = test_inputs[idxs].float()
            accuracy = torch.tensor(0., device=args.device)
            accuracy2 = torch.tensor(0., device=args.device)

            # Test the images for each set of parameters
            for class_id, param in zip(classes, params_all):
                logit = torch.vstack(
                    [model(inp.float().to(args.device), params=param) for inp in torch.split(test_inputs_test, batch)])
                logits.append(logit.squeeze(1).cpu())

            # Compute highest score
            probas = torch.sigmoid(torch.stack(logits))
            predictions = probas.argmax(0)
            predictions2 = []

            # Compute top-2 accuracy
            for iii in range(predictions.shape[0]):
                bb = None
                if predictions[iii].item() != 0:
                    bb = probas[:predictions[iii]]

                if predictions[iii].item() < predictions.shape[0] -1:
                    if bb is not None:
                        bb = torch.cat((bb,probas[(predictions[iii]+1):]), dim=0)
                    else:
                        bb = probas[(predictions[iii]+1):]
                predictions2.append(bb.argmax(0)[0].item())
            pred_label = classes[predictions]
            pred_label2 = classes[predictions2]

            pred_tensor = torch.from_numpy(pred_label).cpu()
            pred_tensor2 = torch.from_numpy(pred_label2).cpu()
            top1 = pred_tensor.eq(test_targets_test.cpu()).float()
            top2 = pred_tensor2.eq(test_targets_test.cpu()).float()
            top2 = torch.logical_or(top1, top2).float()

            accuracy += torch.mean(top1)
            accuracy2 += torch.mean(top2)

            # Save the prediction and labels
            if i == 0:
                pred_label_all = pred_label.tolist()
                test_targets_all = test_targets_test
            else:
                pred_label_all.extend(pred_label.tolist())
                test_targets_all = torch.hstack((test_targets_all,test_targets_test))
            accuracy_print.append(accuracy.item())
            accuracy_print2.append(accuracy2.item())


    save = True
    if save:
        # Save testing accuracies for each shot number
        file_name = os.path.join(args.output_folder, 'Region2_{0}_results_test_'
                                                     '{1}shot_SS{2}_model{3}_NoNCE.csv'.format(region_i, args.num_shots,
                                                                                               step_size, k))
        with open(file_name, 'a') as f:
            write = csv.writer(f)
            write.writerow(accuracy_print)

        # Save top-2 accuracies for each shot number
        file_name = os.path.join(args.output_folder, 'Region2_{0}_results_test_'
                                                '{1}shot_SS{2}_model{3}_NoNCE_top2.csv'.format(region_i,args.num_shots, step_size,k))

        with open(file_name, 'a') as f:
            write = csv.writer(f)
            write.writerow(accuracy_print2)

    # Build confusion matrix
    cf_matrix = confusion_matrix(test_targets_all.tolist(), pred_label_all)
    pred_cm = list(set(pred_label_all))
    GT_cm = list(set(list(set(pred_label_all)) + list(set(test_targets_all.tolist()))))
    cm_sum = cf_matrix.sum(axis=1)[:, np.newaxis]
    for cm_idx in range(len(GT_cm)):
        if cm_sum[cm_idx,0] == 0:
            cm_sum[cm_idx, 0] = 1
    cf_matrix_norm = cf_matrix.astype('float') / cm_sum
    df_cm = pd.DataFrame(cf_matrix_norm, index=[i for i in GT_cm], columns=[i for i in GT_cm])
    plt.figure(figsize=(12, 10))
    plt.title("Confusion matrix for {:}Shots {:}Ways GS{:}".format(args.num_shots, args.num_ways, args.gradient_steps))
    sn.heatmap(df_cm, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xlabel("Predicted class")
    plt.ylabel("Ground truth")
    image_name = os.path.join(args.output_folder, 'Region{:}_ConfusionMatrix_{:}Shots_model{:}_SS{:}_NoNCE.png'.format(region_i, args.num_shots, k, step_size))
    if save:
        plt.savefig(image_name)
    plt.show()


if __name__ == '__main__':

    random.seed(10)

    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--num-shots', type=int, default=4,
                        help='Number of examples per class (k in "k-shot", default: 5).')

    parser.add_argument('--gradient-steps', type=int, default=3)

    parser.add_argument('--num-ways', type=int, default=1,
                        help='Number of classes per task (N in "N-way", default: 4, 1 for one-versus-all).')

    parser.add_argument('--num-cways', type=int, default=4,
                        help='Number of classes per task during contrastive training (N in "N-way", default: 4).')

    parser.add_argument('--first-order', action='store_true', default=True,
                        help='Use the first-order approximation of MAML.')

    parser.add_argument('--step-size', type=float, default=0.32,
                        help='Step-size for the gradient step for adaptation (default: 0.4).')

    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')

    Models_path = os.path.dirname(os.path.dirname(__file__)) + '/models_weights/'

    parser.add_argument('--output-folder', type=str, default=Models_path,
                        help='Path to the output folder for saving the model (optional).')

    #################################################################
    #                           Model to use                        #
    #################################################################
    parser.add_argument('--model', type=str, default='DFC_train',
                        help='Model used for testing. (Implemented: Pretrain, DFC_train, FO_train, Best_DFC, Vanilla_DFC)')

    parser.add_argument('--batch-size', type=int, default=6,
                        help='Number of tasks in a mini-batch of tasks (default: 16).')

    parser.add_argument('--num-batches', type=int, default=300,
                        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
                        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use CUDA if available.')
    parser.add_argument('--set-used', type=str, default='test',
                        help='Use test or train set. To create DFC contrastive tasks: DFC_const, to check them: DFC_read')

    parser.add_argument('--sim_tresh', type=float, default=-1.0,
                        help='Similarity threshold to skip images based on similarity')

    parser.add_argument('--DFC_contrastive_tasks', type=str, default='Initial',
                        help='Predefined option: Initial, Big_SF_bigger, Big_SF_Biggest or the path to a new file')

    args = parser.parse_args()

    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')


    torch.cuda.empty_cache()

    # Create the DFC contrastive task
    if args.set_used == 'DFC_const':
        regions_shuffle = [0,1,2,3,4,5,6]
        random.shuffle(regions_shuffle)
        # Number of tasks: default 30, more task 40
        nbr_tasks = 40
        # Safety Margin between tasks. Default 3, increased 5
        SF_range = 5
        for region_id in regions_shuffle:
            DFC_tasks(args, region_id, nbr_tasks, SF_range, True)

    # Check DFC contrastive tasks
    if args.set_used == 'DFC_read':
        region_id = 0
        tasks = read_DFC_cont(num_ways=args.num_cways, file_name=args.DFC_contrastive_tasks)

    shots_possibility = [1,5,10]

    output_folders = [Models_path]


    DFC_lr = 0.005
    DFC_ss = 0.25
    FO_lr = 5e-3
    FO_ss = 0.285
    # Choose dataset to use: either DFC or FO
    dataset = 'DFC'

    FO_region_testing = 7
    if args.set_used != 'DFC_const':
        if dataset == 'DFC':
            for region_id in range(len(regions_DFC)):
                for nbr_shots in range(len(shots_possibility)):

                    if args.set_used == 'train':

                        # Local parameters number of gradient steps
                        args.gradient_steps = 5

                        # Number of images per contrastive classes
                        args.num_shots = 10

                        # Number of ways, for one versus all 1
                        args.num_ways = 1

                        # Which DFC contrastive file to use: Initial, Big_SF_bigger, Big_SF_Biggest
                        args.DFC_contrastive_tasks = 'Initial'

                        train_oneversusall_DFC_pred(args, learning_rate=DFC_lr, step_size=DFC_ss)

                    if args.set_used == 'test':

                        # Trained model to load. Can be DFC_train, Best_DFC, Vanilla_DFC, FO_train and pretrain
                        args.model = 'Best_DFC'

                        # Number of shots at testing
                        args.num_shots = shots_possibility[nbr_shots]

                        # Local update number of gradient steps
                        args.gradient_steps = 20

                        torch.cuda.empty_cache()

                        # Number of class present
                        args.num_ways = 4

                        test_oneversusall_DFC_pred(args, DFC_lr, DFC_ss, region_i = region_id)
                    if args.set_used == 'train':
                        break
                if args.set_used == 'train':
                    break
        else:
            print("This dataset has not been implemented")


