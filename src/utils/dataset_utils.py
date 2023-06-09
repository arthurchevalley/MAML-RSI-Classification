import random

import numpy as np
import csv

import pandas as pd
from torch.utils.data import Dataset
from glob import glob
import rasterio
import torch
from tqdm import tqdm
import os

np.random.seed(0)
torch.manual_seed(0)

IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water"])


regions_FO = ["accra_20181031",
                   "biscay_20180419",
                   "danang_20181005",
                   "lagos_20200505",
                   "mandaluyong_20180314",
                   "neworleans20200202",
                   "panama_20190415",
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


import numpy as np
from skimage import exposure

def get_rgb(s2):

    s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]

    rgb_idx = [s2bands.index(b) for b in np.array(['S2B4', 'S2B3', 'S2B2'])]

    X = np.clip(s2, a_min=0, a_max=1)

    rgb = np.swapaxes(X[rgb_idx, :, :], 0, 2)
    # rgb = exposure.rescale_intensity(rgb)
    rgb = exposure.equalize_hist(rgb)
    #rgb = exposure.equalize_adapthist(rgb, clip_limit=0.1)
    # rgb = exposure.adjust_gamma(rgb, gamma=0.8, gain=1)

    rgb *= 255

    return rgb

def split_support(ds, shots):

    classes, counts = np.unique(ds.index.maxclass, return_counts=True)
    classes = classes[counts > 2 * shots]

    supports = []
    for c in classes:
        samples = ds.index.loc[ds.index.maxclass == c].reset_index()
        support = samples.sample(shots)
        supports.append(support)

    return pd.concat(supports)

l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

class MetaDataset(Dataset):
    def __init__(self, root, transform=lambda x: x, set='test'):
        super(Dataset).__init__()
        self.root = root
        self.transform = transform

        files = glob(os.path.join(root, "*", "*", "*", "*.tif"))
        files_test = glob(os.path.join(root, "*","task_0", "*", "*.tif"))

        if set == 'train':# or set == 'test':
            for i in range(len(files_test)):
                files.remove(files_test[i])
        elif set == 'test':
            files.clear()
            files = files_test.copy()

        files = [f.replace(root+"/", "") for f in files]
        region, task, classname, imagename = list(zip(*[f.split("/") for f in files]))

        self.index = pd.DataFrame([region, task, classname, imagename, files],
                             index=["region", "task", "classname", "imagename", "files"]).T.set_index(
            ["region", "task"])
        msk = self.index.groupby(["region", "task"]).count().classname == 40
        self.index = self.index.loc[msk]


        self.tasks = self.index.index.unique()
        # performance optim
        self.index = self.index.sort_index()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task = self.tasks[item]
        samples = self.index.loc[task]
        classes = list(samples.classname.unique())

        is_support = np.array([bool(i % 2) for i in range(len(samples))])
        labels = np.array([classes.index(c) for c in samples.classname])

        data = []
        for f in samples.files:
            with rasterio.open(os.path.join(self.root, f), "r") as src:
                data.append(src.read())
        data = np.stack(data)

        # if l1c image, use same bands as l2a
        if data.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            data = data[:, idx]

        y_support = labels[is_support].astype(np.int16)
        x_support = self.transform(data[is_support])
        y_query = labels[~is_support].astype(np.int16)
        x_query = self.transform(data[~is_support])

        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]

        return x_support, y_support, x_query, y_query

class MetaDataset_Increase_Triple(Dataset):
    """
    Floating object dataset class double for more data augmentation
    """
    def __init__(self, root, transform=lambda x: x, set='test'):
        super(Dataset).__init__()
        self.root = root
        self.transform = transform

        files = glob(os.path.join(root, "*", "*", "*", "*.tif"))
        files_test = glob(os.path.join(root, "*","task_0", "*", "*.tif"))

        if set == 'train':
            for i in range(len(files_test)):
                files.remove(files_test[i])
        elif set == 'test':
            files.clear()
            files = files_test.copy()

        files = [f.replace(root+"/", "") for f in files]
        region, task, classname, imagename = list(zip(*[f.split("/") for f in files]))

        region = list(region)
        task = list(task)
        classname = list(classname)
        imagename = list(imagename)

        region2 = region.copy()
        region3 = region.copy()

        task2 = task.copy()
        task3 = task.copy()

        classname2 = classname.copy()
        classname3 = classname.copy()

        imagename2 = imagename.copy()
        imagename3 = imagename.copy()

        files2 = files.copy()
        files3 = files.copy()


        region2[:] = map(lambda x: 'D'+x, region2)
        region3[:] = map(lambda x: 'DD'+x, region3)

        region = tuple(region+region2+region3)
        task = tuple(task+task2+task3)
        classname = tuple(classname + classname2+classname3)
        imagename = tuple(imagename + imagename2+classname3)
        files = files + files2 + files3
        self.index = pd.DataFrame([region, task, classname, imagename, files],
                             index=["region", "task", "classname", "imagename", "files"]).T.set_index(
            ["region", "task"])
        msk = self.index.groupby(["region", "task"]).count().classname == 40
        self.index = self.index.loc[msk]


        self.tasks = self.index.index.unique()

        # performance optim
        self.index = self.index.sort_index()


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task = self.tasks[item]
        samples = self.index.loc[task]
        classes = list(samples.classname.unique())

        is_support = np.array([bool(i % 2) for i in range(len(samples))])
        labels = np.array([classes.index(c) for c in samples.classname])
        data = []

        for f in samples.files:
            if f[1] == 'D':
                f = f[2:]
            elif f[0] == 'D':
                f = f[1:]

            with rasterio.open(os.path.join(self.root, f), "r") as src:
                data.append(src.read())
        data = np.stack(data)

        # if l1c image, use same bands as l2a
        if data.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            data = data[:, idx]

        y_support = labels[is_support].astype(np.int16)
        x_support = self.transform(data[is_support])
        y_query = labels[~is_support].astype(np.int16)
        x_query = self.transform(data[~is_support])

        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]

        return x_support, y_support, x_query, y_query

class MetaDataset_Increase_Double(Dataset):
    """
    Floating object dataset class triple for more data augmentation
    """
    def __init__(self, root, transform=lambda x: x, set='test', testing_id = None):
        super(Dataset).__init__()
        self.root = root
        self.transform = transform

        if testing_id is None:
            idx = random.randint(0, len(regions_FO)-1)
        else:
            idx = testing_id

        files = glob(os.path.join(root, "*.tif"))
        test_file_name = regions_FO[idx] + ".tif"

        files_test = glob(os.path.join(root, test_file_name))

        if set == 'train':
            for i in range(len(files_test)):
                files.remove(files_test[i])
        elif set == 'test':
            files.clear()
            files = files_test.copy()
        print(files)
        files = [f.replace(root+"/", "") for f in files]
        region, task, classname, imagename = list(zip(*[f.split("/") for f in files]))

        region = list(region)
        task = list(task)
        classname = list(classname)
        imagename = list(imagename)

        # Data augmentation by doubling the data
        region2 = region.copy()
        task2 = task.copy()
        classname2 = classname.copy()
        imagename2 = imagename.copy()
        files2 = files.copy()
        region2[:] = map(lambda x: 'D'+x, region2)

        region = tuple(region+region2)
        task = tuple(task+task2)
        classname = tuple(classname + classname2)
        imagename = tuple(imagename + imagename2)
        files = files + files2
        self.index = pd.DataFrame([region, task, classname, imagename, files],
                             index=["region", "task", "classname", "imagename", "files"]).T.set_index(
            ["region", "task"])
        msk = self.index.groupby(["region", "task"]).count().classname == 40
        self.index = self.index.loc[msk]

        self.tasks = self.index.index.unique()
        # performance optim
        self.index = self.index.sort_index()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task = self.tasks[item]
        samples = self.index.loc[task]
        classes = list(samples.classname.unique())

        is_support = np.array([bool(i % 2) for i in range(len(samples))])
        labels = np.array([classes.index(c) for c in samples.classname])

        data = []
        for f in samples.files:
            with rasterio.open(os.path.join(self.root, f), "r") as src:
                data.append(src.read())
        data = np.stack(data)

        # if l1c image, use same bands as l2a
        if data.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            data = data[:, idx]

        y_support = labels[is_support].astype(np.int16)
        x_support = self.transform(data[is_support])
        y_query = labels[~is_support].astype(np.int16)
        x_query = self.transform(data[~is_support])

        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]
        return x_support, y_support, x_query, y_query


class MetaDataset_OVA(Dataset):
    """
    Floating object dataset class tripled for more data augmentation and one versus all approach
    """
    def __init__(self, root, transform=lambda x: x, set='test', num_ways = 4, eval_classes = 0):
        super(Dataset).__init__()
        self.root = root
        self.transform = transform
        self.classes = []
        self.num_ways = num_ways
        self.eval_classes = eval_classes

        idx = random.randint(0, len(regions_FO) - 1)
        idx = 7 #pour southafrica
        print("The region for testing is {:}".format(regions_FO[idx]))
        files = glob(os.path.join(root, "*", "*", "*", "*.tif"))
        files_test = glob(os.path.join(root, regions_FO[idx], "*", "*", "*.tif"))
        if set == 'train':
            for i in range(len(files_test)):
                files.remove(files_test[i])
        elif set == 'test':
            files.clear()
            files = files_test.copy()

        files = [f.replace(root+"/", "") for f in files]
        region, task, classname, imagename = list(zip(*[f.split("/") for f in files]))

        region = list(region)
        task = list(task)
        classname = list(classname)
        imagename = list(imagename)

        region2 = region.copy()
        region3 = region.copy()

        task2 = task.copy()
        task3 = task.copy()

        classname2 = classname.copy()
        classname3 = classname.copy()

        imagename2 = imagename.copy()
        imagename3 = imagename.copy()

        files2 = files.copy()
        files3 = files.copy()


        region2[:] = map(lambda x: 'D'+x, region2)
        region3[:] = map(lambda x: 'DD'+x, region3)

        region = tuple(region+region2+region3)
        task = tuple(task+task2+task3)
        classname = tuple(classname + classname2+classname3)
        imagename = tuple(imagename + imagename2+classname3)

        files = files + files2 + files3
        self.index = pd.DataFrame([region, task, classname, imagename, files],
                             index=["region", "task", "classname", "imagename", "files"]).T.set_index(
            ["region", "task"])
        msk = self.index.groupby(["region", "task"]).count().classname == 40
        self.index = self.index.loc[msk]


        self.tasks = self.index.index.unique()

        # performance optim
        self.index = self.index.sort_index()


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task = self.tasks[item]
        samples = self.index.loc[task]
        self.classes = list(samples.classname.unique())

        samples = samples.loc[samples['classname'].isin(self.classes)]


        is_support = np.array([bool(i % 2) for i in range(len(samples))])
        labels = np.array([self.classes.index(c) for c in samples.classname])

        data = []

        for f in samples.files:
            if f[1] == 'D':
                f = f[2:]
            elif f[0] == 'D':
                f = f[1:]

            with rasterio.open(os.path.join(self.root, f), "r") as src:
                data.append(src.read())
        data = np.stack(data)

        # if l1c image, use same bands as l2a
        if data.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            data = data[:, idx]

        y_support = labels[is_support].astype(np.int16)
        x_support = self.transform(data[is_support])
        y_query = labels[~is_support].astype(np.int16)
        x_query = self.transform(data[~is_support])

        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]

        return x_support, y_support, x_query, y_query

class MetaDataset_OVA_ratio(Dataset):
    """
    Floating object dataset class tripled for more data augmentation and with the same structure as the DFC dataset
    to reuse histogram similarity functions
    """
    def __init__(self, root, transform=lambda x: x, set='test', num_ways = 4, eval_classes = 0, region_id = None):
        super(Dataset).__init__()
        self.root = root
        self.transform = transform
        self.classes = []
        self.num_ways = num_ways
        self.eval_classes = eval_classes

        if region_id is None:
            idx = random.randint(0, len(regions_FO) - 1)
        else:
            idx = region_id
        files = glob(os.path.join(root, "*", "*", "*", "*.tif"))
        files_test = glob(os.path.join(root, regions_FO[idx], "*", "*", "*.tif"))
        if set == 'train':
            for i in range(len(files_test)):
                files.remove(files_test[i])
        elif set == 'test':
            files.clear()
            files = files_test.copy()

        files = [f.replace(root+"/", "") for f in files]
        region, task, classname, imagename = list(zip(*[f.split("/") for f in files]))

        region = list(region)
        task = list(task)
        classname = list(classname)
        imagename = list(imagename)

        region2 = region.copy()
        region3 = region.copy()

        task2 = task.copy()
        task3 = task.copy()

        classname2 = classname.copy()
        classname3 = classname.copy()

        imagename2 = imagename.copy()
        imagename3 = imagename.copy()

        files2 = files.copy()
        files3 = files.copy()


        region2[:] = map(lambda x: 'D'+x, region2)
        region3[:] = map(lambda x: 'DD'+x, region3)

        region = tuple(region+region2+region3)
        task = tuple(task+task2+task3)
        classname = tuple(classname + classname2+classname3)
        imagename = tuple(imagename + imagename2+classname3)
        maxclass = []
        index = []
        for i in range(len(classname)):
            if classname[i] == 'class_0':
                to_add = 0
            elif classname[i] == 'class_1':
                to_add = 1
            elif classname[i] == 'class_2':
                to_add = 2
            elif classname[i] == 'class_3':
                to_add = 3
            index.append(i)
            maxclass.append(to_add)
        maxclass = tuple(maxclass)
        index = tuple(index)
        files = files + files2 + files3
        self.index = pd.DataFrame([region, task, classname, imagename, files, maxclass, index],
                             index=["region", "task", "classname", "imagename", "files", "maxclass","index"]).T.set_index(
            ["region", "task"])
        msk = self.index.groupby(["region", "task"]).count().classname == 40
        self.index = self.index.loc[msk]


        self.tasks = self.index.index.unique()

        # performance optim
        self.index = self.index.sort_index()


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task = self.tasks[item]
        samples = self.index.loc[task]
        self.classes = list(samples.classname.unique())

        samples = samples.loc[samples['classname'].isin(self.classes)]


        is_support = np.array([bool(i % 2) for i in range(len(samples))])
        labels = np.array([self.classes.index(c) for c in samples.classname])

        data = []

        for f in samples.files:
            if f[1] == 'D':
                f = f[2:]
            elif f[0] == 'D':
                f = f[1:]

            with rasterio.open(os.path.join(self.root, f), "r") as src:
                data.append(src.read())
        data = np.stack(data)

        # if l1c image, use same bands as l2a
        if data.shape[1] == 13:
            idx = np.array([l1cbands.index(b) for b in l2abands])
            data = data[:, idx]

        y_support = labels[is_support].astype(np.int16)
        x_support = self.transform(data[is_support])
        y_query = labels[~is_support].astype(np.int16)
        x_query = self.transform(data[~is_support])

        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]

        return x_support, y_support, x_query, y_query

class DFCDataset(Dataset):
    def __init__(self, dfcpath, region, transform):
        super(DFCDataset, self).__init__()
        self.dfcpath = dfcpath

        indexfile = os.path.join(dfcpath, "index.csv")
        self.transform = transform

        if os.path.exists(indexfile):
            index = pd.read_csv(indexfile)
        else:

            tifs = glob(os.path.join(dfcpath, "*/dfc_*/*.tif"))
            assert len(tifs) > 1

            index_dict = []
            for t in tqdm(tifs):
                basename = os.path.basename(t)
                path = t.replace(dfcpath, "")

                # remove leading slash if exists
                path = path[1:] if path.startswith("/") else path

                seed, season, type, region, tile = basename.split("_")

                with rasterio.open(os.path.join(dfcpath, path), "r") as src:
                    arr = src.read()

                classes, counts = np.unique(arr, return_counts=True)

                maxclass = classes[counts.argmax()]

                N_pix = len(arr.reshape(-1))
                counts_ratio = counts / N_pix

                # multiclass labelled with at least 10% of occurance following Schmitt and Wu. 2021
                multi_class = classes[counts_ratio > 0.1]
                multi_class_fractions = counts_ratio[counts_ratio > 0.1]

                s1path = os.path.join(f"{seed}_{season}", f"s1_{region}", basename.replace("dfc", "s1"))
                assert os.path.exists(os.path.join(dfcpath, s1path))

                s2path = os.path.join(f"{seed}_{season}", f"s2_{region}", basename.replace("dfc", "s2"))
                assert os.path.exists(os.path.join(dfcpath, s2path))

                lcpath = os.path.join(f"{seed}_{season}", f"lc_{region}", basename.replace("dfc", "lc"))
                assert os.path.exists(os.path.join(dfcpath, lcpath))

                index_dict.append(
                    dict(
                        basename=basename,
                        dfcpath=path,
                        seed=seed,
                        season=season,
                        region=region,
                        tile=tile,
                        maxclass=maxclass,
                        multi_class=multi_class,
                        multi_class_fractions=multi_class_fractions,
                        s1path=s1path,
                        s2path=s2path,
                        lcpath=lcpath
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)

        index = index.reset_index()
        self.index = index.set_index(["region", "season"])
        self.index = self.index.sort_index()
        self.index = self.index.loc[region]
        ratio = [[x, (self.index.loc[region,'maxclass'].tolist()).count(x)] for x in set(self.index.loc[region,'maxclass'].tolist())]
        self.ratio_all_classes = [0,0,0,0,0,0,0,0,0,0,0]
        for i in range(11):
            if i in set(self.index.loc[region,'maxclass'].tolist()):
                self.ratio_all_classes[i] = (self.index.loc[region,'maxclass'].tolist()).count(i)

        self.region_seasons = self.index.index.unique().tolist()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.loc[self.index["index"] == item].iloc[0]

        with rasterio.open(os.path.join(self.dfcpath, row.s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
            lc = src.read(1)

        input, target = self.transform(s1, s2, lc)

        # TODO change comment
        s = False
        if s:
            input_np = input.numpy()
            tmp1 = input_np[0:12,:,:].copy()
            input = tmp1
            input = torch.from_numpy(input)

        return input.float(), target

class DFCDataset_contrastive(Dataset):
    def __init__(self, dfcpath, region, transform, constractive_ids = None, num_ways = 4):
        super(DFCDataset_contrastive, self).__init__()
        self.dfcpath = dfcpath

        indexfile = os.path.join(dfcpath, "index.csv")
        self.transform = transform

        if os.path.exists(indexfile):
            print(f"loading {indexfile}")
            index = pd.read_csv(indexfile)
        else:

            tifs = glob(os.path.join(dfcpath, "*/dfc_*/*.tif"))
            assert len(tifs) > 1

            index_dict = []
            for t in tqdm(tifs):
                basename = os.path.basename(t)
                path = t.replace(dfcpath, "")

                # remove leading slash if exists
                path = path[1:] if path.startswith("/") else path

                seed, season, type, region_name, tile = basename.split("_")

                with rasterio.open(os.path.join(dfcpath, path), "r") as src:
                    arr = src.read()

                classes, counts = np.unique(arr, return_counts=True)

                maxclass = classes[counts.argmax()]

                N_pix = len(arr.reshape(-1))
                counts_ratio = counts / N_pix

                # multiclass labelled with at least 10% of occurance following Schmitt and Wu. 2021
                multi_class = classes[counts_ratio > 0.1]
                multi_class_fractions = counts_ratio[counts_ratio > 0.1]

                s1path = os.path.join(f"{seed}_{season}", f"s1_{region_name}", basename.replace("dfc", "s1"))
                assert os.path.exists(os.path.join(dfcpath, s1path))

                s2path = os.path.join(f"{seed}_{season}", f"s2_{region_name}", basename.replace("dfc", "s2"))
                assert os.path.exists(os.path.join(dfcpath, s2path))

                lcpath = os.path.join(f"{seed}_{season}", f"lc_{region_name}", basename.replace("dfc", "lc"))
                assert os.path.exists(os.path.join(dfcpath, lcpath))

                index_dict.append(
                    dict(
                        basename=basename,
                        dfcpath=path,
                        seed=seed,
                        season=season,
                        region=region_name,
                        tile=tile,
                        maxclass=maxclass,
                        multi_class=multi_class,
                        multi_class_fractions=multi_class_fractions,
                        s1path=s1path,
                        s2path=s2path,
                        lcpath=lcpath
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)

        index = index.reset_index()
        self.index = index.set_index(["region", "season"])
        self.index = self.index.sort_index()
        self.index = self.index.loc[region]
        ratio = [[x, (self.index.loc[region,'maxclass'].tolist()).count(x)] for x in set(self.index.loc[region,'maxclass'].tolist())]
        self.ratio_all_classes = [0,0,0,0,0,0,0,0,0,0,0]
        for i in range(11):
            if i in set(self.index.loc[region,'maxclass'].tolist()):
                self.ratio_all_classes[i] = (self.index.loc[region,'maxclass'].tolist()).count(i)
        self.index['Task'] = pd.NaT
        self.index['Contrastive_Class'] = pd.NaT
        self.index['item_idxs'] = pd.NaT
        self.multitask = 1
        self.past_index = 0
        if constractive_ids is not None:
            nbr_task = constractive_ids.shape[0]
            random.shuffle(constractive_ids)
            for i in range(nbr_task):
                task = constractive_ids[i,:]
                for j in range(num_ways):
                    for k in range(len(task[j])):
                        idx = task[j][k]
                        self.index.loc[self.index['index'] == idx, ['Task','Contrastive_Class']] = i, j
                        self.index.loc[self.index['index'] == idx, ['item_idxs']] = self.past_index
                        self.past_index = self.past_index + 1
                tasks_to_add = self.index[self.index['Task'] == i]
                if i == 0:
                    tasks = tasks_to_add
                else:
                    tasks = pd.concat([tasks,tasks_to_add])

            self.tasks = tasks.sort_values(by=['Task','Contrastive_Class'],ascending=True)
            self.region_seasons = self.index.index.unique().tolist()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):

        item_task = self.tasks.loc[self.tasks["item_idxs"]==item]["Task"][-1]
        min_tasks = self.tasks.pivot_table(index=["Task"], aggfunc='size').min()

        rows = self.tasks.loc[self.tasks["Task"] == item_task]
        labels = []
        nbr_rows = random.sample([i for i in range(int(rows.size/15))],min_tasks)
        iter = 0
        for nbr_row in nbr_rows:
            row = rows.iloc[nbr_row]
            labels.append(row.Contrastive_Class)
            with rasterio.open(os.path.join(self.dfcpath, row.s1path), "r") as src:
                s1 = src.read()

            with rasterio.open(os.path.join(self.dfcpath, row.s2path), "r") as src:
                s2 = src.read()

            with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
                lc = src.read(1)
            if iter == 0:
                iter += 1
                inputs, _ = self.transform(s1, s2, lc)
            elif iter == 1:
                iter += 1
                new_inputs, _ = self.transform(s1, s2, lc)
                inputs = torch.stack([inputs, new_inputs], dim=0)
            else:
                new_inputs, _ = self.transform(s1, s2, lc)
                new_inputs = new_inputs.unsqueeze(dim=0)
                inputs = torch.cat([inputs, new_inputs], dim=0)
        is_support = np.array([bool(i % 2) for i in range(len(inputs))])
        labels = np.asarray(labels)
        y_support = labels[is_support].astype(np.int16)
        x_support = inputs[is_support].numpy()
        y_query = labels[~is_support].astype(np.int16)
        x_query = inputs[~is_support].numpy()
        x_support, y_support, x_query, y_query = [torch.from_numpy(t) for t in [x_support, y_support, x_query, y_query]]

        return x_support, y_support, x_query, y_query

    def contrastive_classes(self,region, SF_range):

        to_add_classes = [0, 1]
        safety_range = SF_range
        items = self.index.loc[region,"index"].tolist()
        cclass = pd.DataFrame([],columns=['Bounds','center X','center Y','idx'])
        for item in items:
            row = self.index.loc[self.index["index"] == item].iloc[0]
            with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
                center = src.xy(src.height // 2, src.width // 2)
                cclass_new = pd.DataFrame({'Bounds':[[src.bounds.top,src.bounds.bottom,src.bounds.left,src.bounds.right]], 'center X':[center[0]], 'center Y':[center[1]],'idx':[item]}, index=[item])
                cclass.reset_index()
                cclass_new.reset_index()
                cclass = pd.concat([cclass, cclass_new], axis = 0)
        max_x = cclass[['center X']].max()[0]
        max_y = cclass[['center Y']].max()[0]
        min_x = cclass[['center X']].min()[0]
        min_y = cclass[['center Y']].min()[0]
        for multitask in range(self.multitask):
            classes_id = [[], [], [], []]
            if (multitask%3) == 0:
                items_copy = items.copy()
                items_center = items.copy()
            for class_nbr in range(4):
                while True:
                    idx = random.sample(items_center,1)[0]
                    center_values = (cclass[['center X','center Y','idx']].loc[idx]).tolist()
                    borders = (cclass[['Bounds']].loc[idx]).tolist()[0]
                    border = abs(borders[0]-borders[1])
                    if (min_x < center_values[0] < max_x) and (min_y < center_values[1] < max_y):
                        break
                for i in range(-safety_range,safety_range+1):
                    for j in range(-safety_range, safety_range+1):
                        height_x = center_values[0] + i * border
                        width_y = center_values[1] + j * border

                        to_add = ((cclass.loc[(cclass['center X'] == height_x) & (cclass['center Y'] == width_y)])[
                            'idx']).tolist()
                        if len(to_add) == 0 or to_add[0] not in items_copy:
                            continue
                        else:
                            if abs(i) in to_add_classes and abs(j) in to_add_classes:
                                classes_id[class_nbr].append(to_add[0])
                                items_copy.remove(to_add[0])
                            if to_add[0] in items_center:
                                items_center.remove(to_add[0])
                while len(classes_id[class_nbr]) < 9:
                    classes_id[class_nbr].append(-1)

            if multitask == 0:
                classes_id_multitasks = classes_id
            else:
                classes_id_multitasks = np.concatenate((classes_id_multitasks,classes_id))

        return classes_id_multitasks


class FODataset_contrastive(Dataset):
    def __init__(self, path, region, transform, constractive_ids = None, num_ways = 4) -> None:
        super(FODataset_contrastive).__init__()
        
        self.path = path
        self.transform = transform
        indexfile = os.path.join(path, "index.csv")

        if os.path.exists(indexfile):
            print(f"loading {indexfile}")
            index = pd.read_csv(indexfile)
        else:

            tifs = glob(os.path.join(path, "base_image/*.tif"))
            assert len(tifs) > 1

            index_dict = []
            for t in tqdm(tifs):
                
                basename = os.path.basename(t)
                if 'l2a' in basename:
                    continue
                #path = t.replace(path, "")

                # remove leading slash if exists
                #path = path[1:] if path.startswith("/") else path

                region_name, date_time = basename.split("_")

                #with rasterio.open(t, "r") as src:
                #    arr = src.read()

                #if arr.shape[0] == 13:
                #    idx = np.array([l1cbands.index(b) for b in l2abands])
                #    arr = arr[idx]

                index_dict.append(
                    dict(
                        basename=basename[:-4],
                        path=t,
                        region=region_name,
                        date_time = date_time[:-4]
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)
            
        index = index.reset_index()
        self.index = index.set_index(["basename", "date_time"])
        self.index = self.index.sort_index()
        if region is not None:
            self.index = self.index.loc[region]
                
        self.index['Task'] = pd.NaT
        self.index['Contrastive_Class'] = pd.NaT
        self.index['item_idxs'] = pd.NaT
        self.multitask = 1
        self.past_index = 0
        
    def contrastive_classes(self,regions, SF_range, file_name, num_cway=4):

        to_add_classes = [0, 1]
        safety_range = SF_range
        if regions is None:
            regions = regions_FO
        elif type(regions) == list:
            pass
        else:
            regions = [regions]
            
        for region in regions:
            items = self.index.loc[region, 'index']
            items = items.tolist()
            cclass = pd.DataFrame([],columns=['Bounds','center X','center Y','idx'])
            for item in items:
                row = self.index.loc[self.index["index"] == item].iloc[0]
                with rasterio.open(row.path, "r") as src:
                    center = src.xy(src.height // 2, src.width // 2)
                    cclass_new = pd.DataFrame({'Bounds':[[src.bounds.top,src.bounds.bottom,src.bounds.left,src.bounds.right]], 'center X':[center[0]], 'center Y':[center[1]],'idx':[item]}, index=[item])
                    cclass.reset_index()
                    cclass_new.reset_index()
                    cclass = pd.concat([cclass, cclass_new], axis = 0)
            max_x = cclass[['center X']].max()[0]
            max_y = cclass[['center Y']].max()[0]
            min_x = cclass[['center X']].min()[0]
            min_y = cclass[['center Y']].min()[0]
            for multitask in range(self.multitask):
                classes_id = [[], [], [], []]
                if (multitask%3) == 0:
                    items_copy = items.copy()
                    items_center = items.copy()
                for class_nbr in range(num_cway):
                    while True:
                        idx = random.sample(items_center,1)[0]
                        center_values = (cclass[['center X','center Y','idx']].loc[idx]).tolist()
                        borders = (cclass[['Bounds']].loc[idx]).tolist()[0]
                        border = abs(borders[0]-borders[1])
                        if (min_x < center_values[0] < max_x) and (min_y < center_values[1] < max_y):
                            break
                    for i in range(-safety_range,safety_range+1):
                        for j in range(-safety_range, safety_range+1):
                            height_x = center_values[0] + i * border
                            width_y = center_values[1] + j * border

                            to_add = ((cclass.loc[(cclass['center X'] == height_x) & (cclass['center Y'] == width_y)])[
                                'idx']).tolist()
                            if len(to_add) == 0 or to_add[0] not in items_copy:
                                continue
                            else:
                                if abs(i) in to_add_classes and abs(j) in to_add_classes:
                                    classes_id[class_nbr].append(to_add[0])
                                    items_copy.remove(to_add[0])
                                if to_add[0] in items_center:
                                    items_center.remove(to_add[0])
                    while len(classes_id[class_nbr]) < 9:
                        classes_id[class_nbr].append(-1)

                if multitask == 0:
                    classes_id_multitasks = classes_id
                else:
                    classes_id_multitasks = np.concatenate((classes_id_multitasks,classes_id))

            with open(file_name, 'a', newline='') as f:
                write = csv.writer(f)
                write.writerows(classes_id_multitasks)