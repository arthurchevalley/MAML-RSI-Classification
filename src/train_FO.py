import csv

from utils.dataset_utils import MetaDataset_Increase_Double,\
    MetaDataset_OVA_ratio, FODataset_contrastive
from torch.utils.data import DataLoader

import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
import logging
from collections import OrderedDict


import torch
import numpy as np
import random


from utils_learning import *

logger = logging.getLogger(__name__)

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

def FO_tasks(args,nbr_tasks,SF_range, all = False, region_id=None):
    """
    Create the contrastive DFC tasks
    :param args: arg parser provided by the user
    :param region_id: Region on which the tasks are created
    :param nbr_tasks: Number of tasks
    :param SF_range: Safety margin around a given class
    :param all: Bool to indicate if all region are used to create tasks
    """

    path = os.path.dirname(os.path.dirname(__file__)) + '/DATA/FO_DATA/'
    
    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:, :, r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x*1e-4

    ds = FODataset_contrastive(path, region=region_id, transform=transform)

    # Tasks list file name
    if all:
        file_name = os.path.join(path+'FO_tasks/',
                                 '{:}tasks_{:}cways_NCE_all_{:}MT_all_bigSF.csv'.format(nbr_tasks, args.num_cways, ds.multitask))
    else:
        file_name = os.path.join(path+'FO_tasks/', '{:}tasks_{:}cways_NCE_all_{:}MT_region{:}.csv'.format(nbr_tasks,args.num_cways,ds.multitask,region_id))

    # Tasks creation and save in the index file
    ds.contrastive_classes(region_id,SF_range, file_name)
        


def read_FO_cont(num_ways = 4, file_name='Initial'):
    """
    Read the DFC contrastive tasks index file
    :param num_ways: Number of contrastive tasks
    :param file_name: File name containing DFC contrastive tasks
    :return: Classes indexes
    """
    path = os.path.dirname(__file__) + '/../DATA/DFC/DFC_Public_Dataset/'

    if file_name == 'Initial':
        file_name = os.path.join(path+'DFC_tasks/', '{:}tasks_{:}cways_NCE_all_{:}MT_all.csv'.format(20, 4, 1))
    elif file_name == 'Big_SF_bigger':
        file_name = os.path.join(path+'DFC_tasks/', '{:}tasks_{:}cways_NCE_all_{:}MT_all_shuffle_bigSF.csv'.format(30, 4, 1))
    elif file_name == 'Big_SF_Biggest':
        file_name = os.path.join(path+'DFC_tasks/', '{:}tasks_{:}cways_NCE_all_{:}MT_all_bigSF.csv'.format(40, 4, 1))
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


def train(args, learning_rate, step_size):
    """
    Training function for the floating object dataset
    :param args: arg parser provided by the user
    :param learning_rate: Training learning rate
    :param step_size: Training step size
    """

    path = os.path.dirname(__file__)[:-3] + 'DATA/FO_DATA/'

    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:, :, r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x*1e-4

    ds = MetaDataset_Increase_Double(path, transform,set=args.set_used, testing_id=0)

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(12,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training accuracy monitoring
    accuracy_print = []
    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            # Images and labels extraction
            train_inputs, train_targets, test_inputs, test_targets = [t.to(device=args.device) for t in batch]
            train_inputs = train_inputs.float()
            test_inputs = test_inputs.float()
            train_targets = train_targets.long()
            test_targets = test_targets.long()
            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)

            # Task training loop
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                model.zero_grad()
                meta_optimizer.zero_grad()

                params = OrderedDict()

                for (name, param) in model.named_parameters():
                    params[name] = param

                # Local Parameters update
                for t in range(args.gradient_steps):
                    train_logit = model(train_input, params=params)
                    inner_loss = F.cross_entropy(train_logit, train_target)
                    model.zero_grad()
                    grads = torch.autograd.grad(inner_loss, params.values(),
                                              create_graph=not args.first_order)

                    params_next = OrderedDict()
                    for (name, param), grad in zip(list(params.items()), grads):
                        params_next[name] = param - step_size * grad
                    params = params_next

                # Outer loss for a task computation
                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            # Model parameters back-propagation for after each batch
            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            accuracy_print.append(accuracy.item())
            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx >= args.num_batches:
                break

    if args.output_folder is not None:
        # Save model
        model_name = os.path.join(args.output_folder, 'maml_contrastiveMAML_'
                                                    '{0}shot_{1}way_lr{2}_ss{3}_oneSA.th'.format(args.num_shots,
                                                                                                 args.num_ways,
                                                                                                 learning_rate, step_size))
        os.makedirs(os.path.dirname(model_name), exist_ok = True)
        with open(model_name, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

        # Save training curve
        file_name = os.path.join(args.output_folder, 'results_train_'
                                                     '{0}shot_{1}way_lr{2}_ss{3}.csv'.format(args.num_shots,
                                                                                             args.num_ways,
                                                                                             learning_rate, step_size))
        with open(file_name, 'w') as f:
            write = csv.writer(f)
            write.writerow(accuracy_print)


def test(args, learning_rate, step_size, region, ss2, lr2):
    """
    Testing function for the floating object dataset. Either one set of adapted parameters for each classes in the
    one versus all case or one set of parameters for the whole model
    :param args: arg parser provided by the user
    :param learning_rate: Training learning rate
    :param step_size: Training step size
    :param region: Region used for testing
    :param ss2: Adaptation step size
    :param lr2: Adaptation learning rate
    :return: Testing accuracies
    """

    path = os.path.dirname(__file__)[:-3] + 'DATA/FO_DATA/'

    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:, :, r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x * 1e-4

    ds = MetaDataset_OVA_ratio(path, transform, set=args.set_used, num_ways=args.num_ways, region_id=region)

    batch = 4 * args.num_shots

    dataloader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=args.num_workers)

    # Model init and load
    model = ConvolutionalNeuralNetwork(12,
                                       4,
                                       hidden_size=args.hidden_size)

    model.to(device=args.device)
    model_file = os.path.join(args.output_folder, 'maml_contrastiveMAML_'
                                                    '{0}shot_{1}way_lr{2}_ss{3}.th'.format(10,
                                                                                           4,
                                                                                           learning_rate,
                                                                                           step_size))

    model.load_state_dict(torch.load(model_file, map_location=args.device))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr2)

    accuracy_print = []

    classes = np.unique(ds.index.maxclass)
    # Nbr of samples to test
    nbr_to_sample = 50
    train_inputs, train_targets, test_inputs, test_targets = split_support_query(ds, args.num_shots,
                                                                                 FO=True,
                                                                                 nbr=nbr_to_sample)
    model.zero_grad()
    train_inputs = train_inputs.float()
    train_targets = train_targets.long()

    nbr_training_samples = train_targets.shape[0]

    model.train()

    params_all = []
    if args.num_ways == 1:
        # Find a local set of parameters for each class
        for target_class in classes:
            model.zero_grad()
            params = OrderedDict(model.meta_named_parameters())
            for t in range(args.gradient_steps):
                # Adaptation images shuffling
                idxs = [i for i in range(train_inputs.shape[0])]
                random.shuffle(idxs)
                train_targets_in = train_targets[idxs]

                # Model evaluation and adaptation
                train_logit = model(train_inputs[idxs].float().to(args.device), params=params)
                inner_loss = F.cross_entropy(train_logit, train_targets_in.to(args.device))

                model.zero_grad()
                grads = torch.autograd.grad(inner_loss, params.values(),
                                            create_graph=not args.first_order)

                # Class parameters update
                params_next = OrderedDict()
                for (name, param), grad in zip(list(params.items()), grads):
                    params_next[name] = param - step_size * grad
                params = params_next

            params_all.append(params)
    else:
        # Find a global set of parameters for the training images
        model.zero_grad()
        meta_optimizer.zero_grad()

        params = OrderedDict(model.meta_named_parameters())
        for (name, param) in model.named_parameters():
            params[name] = param

        # Local adaptation
        for t in range(args.gradient_steps):
            train_logit = model(train_inputs.float().to(args.device), params=params)
            inner_loss = F.cross_entropy(train_logit, train_targets.to(args.device))

            model.zero_grad()
            grads = torch.autograd.grad(inner_loss, params.values(),
                                        create_graph=not args.first_order)

            params_next = OrderedDict()
            for (name, param), grad in zip(list(params.items()), grads):
                params_next[name] = param - ss2 * grad
            params = params_next

    # Adapted model evaluation
    model.eval()
    test_targets_all = []
    pred_label_all = []
    with torch.no_grad():
        # Number of test done
        nbr_tests = 5
        idxs_all = [i for i in range(test_inputs.shape[0])]

        # Number of samples per test
        nbr_to_sample = 50
        for test_idx in range(nbr_tests):
            logits = []
            accuracy = torch.tensor(0., device=args.device)

            if args.num_ways == 1:

                # Randomly select nbr_to_samples images from the test images
                if len(idxs_all) >= nbr_to_sample:
                    idxs = list(random.sample(idxs_all, nbr_to_sample))

                    idxs_all = list(filter(lambda x: x not in set(idxs), idxs_all))

                    test_targets_test = test_targets[idxs].long()
                    test_inputs_test = test_inputs[idxs].float()
                else:
                    test_targets_test = test_targets.long()
                    test_inputs_test = test_inputs.float()

                # Test each local adaptation parameters
                for class_id, param in zip(classes, params_all):
                    logit = torch.vstack(
                        [model(inp.float().to(args.device), params=param) for inp in torch.split(test_inputs_test, batch)])
                    logits.append(logit.squeeze(1).cpu())

                # Find the most confident class as prediction
                probas = torch.sigmoid(torch.stack(logits))
                pred_label = classes[probas.argmax(0)]

                accuracy += get_accuracy_BIN(torch.stack(logits), test_targets_test)
                if test_idx == 0:
                    pred_label_all = pred_label.tolist()
                    test_targets_all = test_targets_test
                else:
                    pred_label_all.extend(pred_label.tolist())
                    test_targets_all = torch.hstack((test_targets_all, test_targets_test))
            else:
                # Randomly select nbr_to_samples images from the test images
                if len(idxs_all) >= nbr_to_sample:
                    idxs = list(random.sample(idxs_all, nbr_to_sample))

                    test_targets_test = test_targets[idxs].long()
                    test_inputs_test = test_inputs[idxs].float()
                else:
                    test_targets_test = test_targets.long()
                    test_inputs_test = test_inputs.float()

                # Test the global parameters adaptation and find the most confident class
                test_logit = model(test_inputs_test.float().to(args.device), params = params)
                probas = torch.sigmoid(test_logit)
                predictions = probas.argmax(1).cpu()
                accuracy += torch.mean(predictions.eq(test_targets_test.cpu()).float())

            accuracy_print.append(accuracy.item())

    # Save testing accuracies
    file_name = os.path.join(args.output_folder, 'OVA_results_test_ratio_'
                                                 '{:}shot_region_{:}.csv'.format(args.num_shots, regions_FO[region]))
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(accuracy_print)

    return np.mean(accuracy_print)


def test_updated(args, learning_rate, step_size, learning_rate_train, step_size_train, testing_region_id=None):
    """
    Updated testing function for the floating object dataset. Allow to test on batch or task images
    :param args: arg parser provided by the user
    :param learning_rate_train: Training learning rate
    :param step_size_train: Training step size
    :param step_size: Adaptation step size
    :param learning_rate: Adaptation learning rate
    :return: Testing accuracies
    """

    path = path = os.path.dirname(__file__)[:-3] + 'DATA/FO_DATA/'

    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:, :, r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x*1e-4

    ds = MetaDataset_Increase_Double(path, transform, set = 'test', testing_id = testing_region_id)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Create and load the model and its parameters
    model = ConvolutionalNeuralNetwork(12,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model_file = os.path.join(args.output_folder, 'maml_contrastiveMAML_'
                                                '{0}shot_{1}way_lr{2}_ss{3}.th'.format(10, 4,
                                                                                       learning_rate_train,
                                                                                       step_size_train))
    model.load_state_dict(torch.load(model_file, map_location=args.device))
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_print = []
    accuracy_print_all = []

    predi_cm = []
    trg_cm = []
    adaptation = True
    # Randomly selected batch and task for training. The number are fixed for repeatability but any could be chosen
    # Task training is initialized if more than 5 shots are tested as the tasks have 5 adaptation image each
    task_training = 3
    if args.num_shots > 5:
        task_training2 = 5
    else:
        task_training2 = task_training
    with tqdm(dataloader, total=args.num_batches) as pbar:
        batch_training = 5
        for batch_idx, batch in enumerate(pbar):
            if batch_idx != batch_training:
                continue
            model.zero_grad()

            train_inputs, train_targets, test_inputs, test_targets = [t.to(device=args.device) for t in batch]

            train_inputs = train_inputs.float()
            test_inputs = test_inputs.float()
            train_targets = train_targets.long()
            test_targets = test_targets.long()

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            accuracy_all = torch.tensor(0., device=args.device)

            # Adaptation of the local parameters
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):

                # Only look at the randomly selected task and batch
                if (task_training != task_idx and task_training2 != task_idx):
                    continue

                # Shuffle the indexes of images used
                idx_shots = []
                if args.num_shots < 5:
                    for ii in range(4):
                        idx = random.randint(ii * 5, 5 * ii + 4)
                        idx_shots.append(idx)
                    train_input = train_input[idx_shots]
                    train_target = train_target[idx_shots]
                else:
                    idx_rnd = [i for i in range(train_target.shape[0])]
                    random.shuffle(idx_rnd)
                    train_input = train_input[idx_rnd]
                    train_target = train_target[idx_rnd]

                model.zero_grad()
                meta_optimizer.zero_grad()

                # Local parameters update
                if adaptation:
                    if task_idx == task_training:
                        params = OrderedDict()
                        for (name, param) in model.named_parameters():
                            params[name] = param

                    # Local parameters update
                    for t in range(args.gradient_steps):
                        train_logit = model(train_input, params=params)
                        inner_loss = F.cross_entropy(train_logit, train_target)
                        model.zero_grad()
                        grads = torch.autograd.grad(inner_loss, params.values(),
                                                  create_graph=not args.first_order)

                        params_next = OrderedDict()
                        for (name, param), grad in zip(list(params.items()), grads):
                            params_next[name] = param-step_size*grad
                        params = params_next

                    if task_idx == task_training2:
                        adaptation = False

                # Testing on already updated parameters
                if not adaptation:
                    test_inputs_all = torch.flatten(test_inputs,  start_dim=0, end_dim=1)
                    test_targets_all = torch.flatten(test_targets,  start_dim=0, end_dim=1)

                    test_logit_all = model(test_inputs_all , params=params)
                    outer_loss += F.cross_entropy(test_logit_all, test_targets_all )
                    _, predictions_all = torch.max(test_logit_all, dim=-1)

                    test_logit = model(test_input, params=params)
                    outer_loss += F.cross_entropy(test_logit, test_target)
                    _, predictions = torch.max(test_logit, dim=-1)

                    predi_cm.extend(predictions.cpu().numpy())
                    trg_cm.extend(test_targets.cpu().numpy())

                    with torch.no_grad():
                        # Accuracy on the remaining images from the task
                        # Accuracy all is the accuracy over a whole batch test images
                        accuracy += get_accuracy(test_logit, test_target)
                        accuracy_all += get_accuracy(test_logit_all, test_targets_all)
                        print("task {:} accuracy: {:}".format(task_idx, get_accuracy(test_logit, test_targets)))
                    torch.cuda.empty_cache()
                    break

            accuracy_print.append(accuracy.item())
            accuracy_print_all.append(accuracy_all.item())
            break

    print("task:",accuracy_print)
    print("batch",accuracy_print_all)

    # Save the testing accuracies
    file_name = os.path.join(args.output_folder, 'results_test_'
                                                '{0}shot_{1}way_lr{2}_ss{3}.csv'.format(args.num_shots, args.num_ways,
                                                                                       learning_rate, step_size))
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(accuracy_print)
        write.writerow(accuracy_print_all)


def train_oneversusall(args, learning_rate, step_size, testing_region):
    """
    Training function for the floating object dataset
    :param args: arg parser provided by the user
    :param learning_rate: Training learning rate
    :param step_size: Training step size
    :param testing_region: Region not used at training and only for testing
    """

    path = os.path.dirname(__file__)[:-3] + 'DATA/FO_DATA/'

    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:,:,r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x*1e-4

    ds = MetaDataset_OVA_ratio(path, transform, set=args.set_used, num_ways=args.num_ways, region_id=testing_region)

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(12,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

    # Similartiy measurment initialisation
    nbr_samples = args.num_shots * args.num_cways
    temperature = 0.5
    loss = Similartiy_class(nbr_samples, args.num_cways, temperature)

    # Training loop
    accuracy_print = [1]
    hist_print = []
    total_skipped = 0
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            # Images and labels extraction
            train_inputs, train_targets, test_inputs, test_targets = [t.to(device=args.device) for t in batch]
            train_inputs = train_inputs.float()
            test_inputs = test_inputs.float()
            train_targets = train_targets.float()
            test_targets = test_targets.float()

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            classes = np.unique(train_targets.cpu())
            skip = 0

            # Task training loop
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                             test_inputs, test_targets)):

                # Random shuffling of training and testing data
                rnd_list = [i for i in range(len(train_target))]
                random.shuffle(rnd_list)
                train_target = train_target[rnd_list]
                train_input = train_input[rnd_list, :, :, :]
                rnd_list = [i for i in range(len(test_target))]
                random.shuffle(rnd_list)
                test_target = test_target[rnd_list]
                test_input = test_input[rnd_list, :, :, :]

                # Random selection of a class for this task
                for c in random.sample(classes.tolist(), args.num_ways):
                    train_target_OVA = (train_target == c).to(float)
                    test_target_OVA = (test_target == c).to(float)


                # Histogram similarity measurement

                # Positive & negatives class indexes
                idx_pos = list(map(lambda x: x > 0, test_target_OVA))
                pos_idx = []
                neg_idx = []
                for id_add, j in enumerate(idx_pos):
                    if j.item():
                        pos_idx.append(id_add)
                    else:
                        neg_idx.append(id_add)

                hist_neg = get_hist(test_input[neg_idx, :, :, :])
                hist_pos = get_hist(test_input[pos_idx, :, :, :])

                # Intra-class similarity
                if hist_pos.shape[1] != 0:
                    intra_class_sim = loss.compute_loss_hist_pos(hist_pos)
                else:
                    intra_class_sim = torch.zeros(1)

                # Inter-class similarity
                if hist_neg.shape[1] != 0:
                    inter_class_sim = loss.compute_loss_hist_negativ(hist_pos, hist_neg, neg_idx)
                else:
                    inter_class_sim = 0.1 * torch.ones(1)

                # Ratio of intra-class over inter-class
                ratio = intra_class_sim / inter_class_sim

                # Skip task if NaN or ratio smaller than 1.0
                if (torch.isnan(ratio)) or (ratio.item() < 1.0):
                    skip += 1
                    total_skipped += 1
                    continue

                model.zero_grad()
                meta_optimizer.zero_grad()

                params = OrderedDict()

                # Local parameter update
                for (name, param) in model.named_parameters():
                    params[name] = param

                for t in range(args.gradient_steps):
                    train_logit = model(train_input, params=params)
                    if args.num_ways == 1:
                        inner_loss = criterion(train_logit.squeeze(1), train_target_OVA)
                    else:
                        inner_loss = F.cross_entropy(train_logit, train_target_OVA)
                    model.zero_grad()
                    grads = torch.autograd.grad(inner_loss, params.values(),
                                              create_graph=not args.first_order)

                    params_next=OrderedDict()
                    for (name, param), grad in zip(list(params.items()), grads):
                        params_next[name] = param - step_size * grad
                    params=params_next

                # Test local parameter to compute outer loss
                test_logit = model(test_input, params=params)
                if args.num_ways == 1:
                    test_logit = test_logit.squeeze(1)
                    outer_loss += F.binary_cross_entropy_with_logits(test_logit, test_target_OVA,
                                                                     weight=intra_class_sim.to(args.device))
                else:
                    outer_loss += F.cross_entropy(test_logit, test_target, weight=intra_class_sim.to(args.device))

                # Compute task accuracy
                with torch.no_grad():
                    if args.num_ways == 1:
                        accuracy += get_accuracy_BIN(test_logit, test_target_OVA)
                    else:
                        accuracy += get_accuracy(test_logit, test_target)

            # Compute outer loss and update model parameters
            # If all task have been skipped, no update
            if skip != args.batch_size:
                outer_loss.div_((args.batch_size-skip))
                accuracy.div_((args.batch_size-skip))
                outer_loss.backward()
                meta_optimizer.step()

                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                accuracy_print.append(accuracy.item())
                hist_print.append(intra_class_sim.item())
            else:
                pbar.set_postfix(accuracy='0.0000')

            if batch_idx >= args.num_batches:
                break

    # Save model
    if args.output_folder is not None:
        # Save training similarity
        hist_file_name = os.path.join(args.output_folder, 'hist_FO.csv')
        with open(hist_file_name, 'w') as f:
            write = csv.writer(f)
            write.writerow(hist_print)

        # Save model weights
        model_name = os.path.join(args.output_folder, 'OVA_contrastiveMAML_vanilla_'
                                                    '{0}shot_{1}way_lr{2}_ss{3}_region{4}.th'.format(args.num_shots,
                                                                                                     args.num_ways,
                                                                                                     learning_rate,
                                                                                                     step_size,
                                                                                                     testing_region))
        os.makedirs(os.path.dirname(model_name), exist_ok = True)
        with open(model_name, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

        # Save training curves
        file_name = os.path.join(args.output_folder, 'OVA_resnet_results_train_ratio_'
                                                     '{0}shot_{1}way_lr{2}_ss{3}.csv'.format(args.num_shots,
                                                                                             args.num_ways,
                                                                                             learning_rate, step_size))
        with open(file_name, 'w') as f:
            write = csv.writer(f)
            write.writerow(accuracy_print)


def test_oneversusall(args, learning_rate, step_size, region):
    """
    Test function for the one versus all approach
    :param args: arg parser provided by the user
    :param learning_rate: Adaptation learning rate
    :param step_size: Adaptation step size
    :param region: Region used for testing
    """
    path = os.path.dirname(__file__)[:-3] + 'DATA/FO_DATA/'

    def transform(x):
        r = random.randint(0, x.shape[3] - 128)
        c = random.randint(0, x.shape[2] - 128)
        x = x[:, :, r:r + 128, c:c + 128]

        if random.random() < 0.5:
            x = x[:, :, ::-1, :]
        return x * 1e-4

    ds = MetaDataset_OVA_ratio(path, transform, set=args.set_used, num_ways=args.num_ways, region_id=region)

    batch = 4 * args.num_shots

    dataloader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=args.num_workers)

    # Define and load model
    model = ConvolutionalNeuralNetwork(12,
                                       1,
                                       hidden_size=args.hidden_size)

    model.to(device=args.device)
    model_file = os.path.join(args.output_folder, 'OVA_contrastiveMAML_vanilla_'
                                                  '{0}shot_{1}way_lr{2}_ss{3}_region7.th'.format(10, 1,
                                                                                                 learning_rate,
                                                                                                 step_size))
    model.load_state_dict(torch.load(model_file, map_location=args.device))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_print = []

    classes = np.unique(ds.index.maxclass)
    nbr_to_sample = 50
    train_inputs, train_targets, test_inputs, test_targets = split_support_query(ds, args.num_shots,
                                                                                 FO=True,
                                                                                 nbr=nbr_to_sample)
    model.zero_grad()
    train_inputs = train_inputs.float()
    train_targets = train_targets.long()

    nbr_training_samples = train_targets.shape[0]

    model.train()

    params_all = []

    # Define a local adaptation for every classes
    for target_class in classes:
        model.zero_grad()

        train_targets_OVA = (train_targets == target_class).to(float)

        params = OrderedDict(model.meta_named_parameters())

        # Adaptation of the local parameters
        for t in range(args.gradient_steps):

            idxs = [i for i in range(train_inputs.shape[0])]
            random.shuffle(idxs)
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

    # Test model for every set of parameters
    model.eval()
    test_targets_all = []
    pred_label_all = []
    with torch.no_grad():
        # Number of batch test
        nbr_tests = 5
        # Number of samples per batch
        nbr_to_sample = 50

        idxs_all = [i for i in range(test_inputs.shape[0])]
        for test_idx in range(nbr_tests):
            logits = []

            # Random selection of nbr_to_sample test images
            if len(idxs_all) >= nbr_to_sample:
                idxs = list(random.sample(idxs_all, nbr_to_sample))

                idxs_all = list(filter(lambda x: x not in set(idxs), idxs_all))

                test_targets_test = test_targets[idxs].long()
                test_inputs_test = test_inputs[idxs].float()
            else:
                test_targets_test = test_targets.long()
                test_inputs_test = test_inputs.float()
            accuracy = torch.tensor(0., device=args.device)

            # Compute the model output for every locally updated parameters
            for class_id, param in zip(classes, params_all):
                logit = torch.vstack(
                    [model(inp.float().to(args.device), params=param) for inp in torch.split(test_inputs_test, batch)])
                logits.append(logit.squeeze(1).cpu())

            # Compute the highest score for the test images
            probas = torch.sigmoid(torch.stack(logits))
            pred_label = classes[probas.argmax(0)]
            accuracy += get_accuracy_BIN(torch.stack(logits),test_targets_test)

            accuracy_print.append(accuracy.item())

    # Save testing accuracies
    file_name = os.path.join(args.output_folder, 'OVA_results_test_ratio_'
                                                 '{:}shot_region_{:}.csv'.format(args.num_shots, regions_FO[region]))
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(accuracy_print)

    return accuracy_print


def main():
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

    Models_path = os.path.dirname(__file__) + '/../models_weights/'

    parser.add_argument('--output-folder', type=str, default=Models_path,
                        help='Path to the output folder for saving the model (optional).')

    #################################################################
    #                           Model to use                        #
    #################################################################
    parser.add_argument('--model', type=str, default='FO_train',
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
    parser.add_argument('--set-used', type=str, default='Create_FO',
                        help='Use test or train set. To create DFC contrastive tasks: DFC_const, to check them: DFC_read')

    parser.add_argument('--sim_tresh', type=float, default=-1.0,
                        help='Similarity threshold to skip images based on similarity')

    parser.add_argument('--DFC_contrastive_tasks', type=str, default='Initial',
                        help='Predefined option: Initial, Big_SF_bigger, Big_SF_Biggest or the path to a new file')

    args = parser.parse_args()

    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')


    torch.cuda.empty_cache()


    shots_possibility = [1,5,10]

    output_folders = [Models_path]

    FO_lr = 5e-3
    FO_ss = 0.285
    # Choose dataset to use: either DFC or FO
    dataset = 'FO'

    FO_region_testing = 7
    if dataset == 'FO':
        if args.set_used == 'Create_FO':
            # Number of tasks: default 30, more task 40
            nbr_tasks = 40
            # Safety Margin between tasks. Default 3, increased 5
            SF_range = 5
            FO_tasks(args, nbr_tasks, SF_range, True)
        else:
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

                    train_oneversusall(args, FO_lr, FO_ss, FO_region_testing)

                if args.set_used == 'test':
                    # Trained model to load
                    args.model = 'pretrained'

                    # Number of shots at testing
                    args.num_shots = shots_possibility[nbr_shots]

                    # Local update number of gradient steps
                    args.gradient_steps = 20

                    torch.cuda.empty_cache()

                    # Number of class present
                    args.num_ways = 4

                    test_updated(args, FO_lr, FO_ss, FO_lr, FO_ss, FO_region_testing)

                if args.set_used == 'train':
                    break

    else:
        print("This dataset has not been implemented")


if __name__ == '__main__':
    main()



