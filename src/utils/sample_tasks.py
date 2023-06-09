import rasterio
import numpy as np
import os
from rasterio.windows import Window
import matplotlib.pyplot as plt
import json
from skimage.exposure import equalize_hist
from tqdm import tqdm
import argparse

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('imagefile', type=str, help='Sentinel 2 imagefile')
    parser.add_argument('tasks_folder', type=str, help='target folder')
    parser.add_argument('--num-ways', type=int, default=4, help='number of classes')
    parser.add_argument('--num-shots', type=int, default=5, help='number of instances per class (the script samples '
                                                                 'twice as many to have enough for query and support '
                                                                 'partitions)')
    parser.add_argument('--radius-between-classes', type=int, default=2000, help='distance between class centers (in pixel)')
    parser.add_argument('--radius_within_class', type=int, default=750, help='distance of samples within one class')
    parser.add_argument('--imagesize', type=int, default=256, help='image size of tiles')


    args = parser.parse_args()

    tasks_folder = args.tasks_folder # "/data/contrastiveMAML/dataset_utils"
    imagefile = args.imagefile #"/data/dataset_utils/data/portalfredSouthAfrica_20180601.tif"
    radius_within_class = args.radius_within_class
    radius_between_classes = args.radius_between_classes
    num_ways = args.num_ways
    num_shots = 2 * args.num_shots
    imagesize = args.imagesize

    with rasterio.open(imagefile, "r") as src:
        rgb = np.array([src.read(bands.index(b)) for b in ["B4", "B3", "B2"]])
        rgb = equalize_hist(rgb).transpose(1, 2, 0)
        invalid_mask = src.read().sum(0) == 0

    for t in tqdm(range(100)):
        taskroot = os.path.join(tasks_folder, os.path.splitext(os.path.basename(imagefile))[0], f"task_{t}")

        if os.path.exists(taskroot):
            print(f"{taskroot} exists. skipping...")
            continue

        try:
            sample_task(t, imagefile, tasks_folder, num_ways, imagesize, radius_between_classes,
                        num_shots, radius_within_class, invalid_mask)
        except ValueError as e:
            print(f"skipping task {t} due to error: {e}")
            continue

        jsonpath = os.path.join(taskroot, "taskinfo.json")

        fig = plot_overview(jsonpath=jsonpath, rgb=rgb)
        fig.savefig(os.path.join(taskroot, "overview.png"), bbox_inches="tight", transparent=False)
        plt.close(fig)

        fig = plot_samples(jsonpath=jsonpath)
        fig.savefig(os.path.join(taskroot, "samples.png"), bbox_inches="tight", transparent=False)
        plt.close(fig)

def sample_task(seed, imagefile, tasks_folder, num_ways, imagesize, radius_between_classes,
                num_shots, radius_within_class, invalid_mask):
    np.random.seed(seed)
    image = os.path.splitext(os.path.basename(imagefile))[0]

    scene_height, scene_width = invalid_mask.shape

    taskroot = os.path.join(tasks_folder, os.path.splitext(os.path.basename(imagefile))[0], f"task_{seed}")

    class_prototypes = sample_class_prototypes(num_class_prototypes=num_ways, imagesize=imagesize,
                                               scene_width=scene_width, scene_height=scene_height,
                                               radius_between_classes=radius_between_classes,
                                               invalid_mask=invalid_mask)

    point_dict_info = []
    for c, (x_proto, y_proto) in enumerate(class_prototypes):

        # sample list of points around the class prototype
        XY = sample_instances_for_prototype(num_instances=num_shots, x_proto=x_proto, y_proto=y_proto,
                                            imagesize=imagesize, invalid_mask=invalid_mask,
                                            radius_within_class=radius_within_class)

        samples = {}
        for i, (x, y) in enumerate(XY):
            window = Window(x - imagesize // 2, y - imagesize // 2, imagesize, imagesize)

            with rasterio.open(imagefile, "r") as src:
                img = src.read(window=window)
                profile = src.profile
                profile["transform"] = src.window_transform(window)
                profile["count"], profile["width"], profile["height"] = img.shape

            relativepath = os.path.join(f"class_{c}", f"{c}-{i}.tif")
            writepath = os.path.join(taskroot, relativepath)
            os.makedirs(os.path.dirname(writepath), exist_ok=True)

            with rasterio.open(writepath, 'w', **profile) as dst:
                dst.write(img)

            samples[str(i)] = {
                "x": x,
                "y": y,
                "path": relativepath
            }

        point_dict_info.append({
            "class": c,
            "x_proto": x_proto,
            "y_proto": y_proto,
            "samples": samples
        })

    taskinfo = {
        "radius_between_classes": radius_between_classes,
        "seed": seed,
        "image": image,
        "radius_within_class": radius_within_class,
        "scene_width": scene_width,
        "scene_height": scene_height,
        "num_ways": num_ways,
        "num_shots": num_shots,
        "imagesize": imagesize,
        "point_dict_info": point_dict_info
    }

    jsonpath = os.path.join(taskroot, "taskinfo.json")
    with open(jsonpath, "w") as f:
        json.dump(taskinfo, f)

def plot_samples(jsonpath):
    with open(jsonpath, "r") as f:
        data = json.load(f)

    taskroot = os.path.dirname(jsonpath)
    image = data["image"]
    seed = data["seed"]
    num_shots = data["num_shots"]
    num_ways = data["num_ways"]

    fig, axs = plt.subplots(num_ways, num_shots, sharey=True, figsize=(2*num_shots, 2*num_ways))
    for ax_row, data in zip(axs, data["point_dict_info"]):
        for i, (ax, (k, sample)) in enumerate(zip(ax_row, data["samples"].items())):
            imagefile = os.path.join(taskroot, sample["path"])
            with rasterio.open(imagefile, "r") as src:
                rgb = np.array([src.read(bands.index(b)) for b in ["B4", "B3", "B2"]])
                rgb = equalize_hist(rgb).transpose(1, 2, 0)
                ax.imshow(rgb)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                if i == 0:
                    ax.set_ylabel(f"class {data['class']}")

    fig.suptitle(f"{image}-{seed} {num_shots}-shot {num_ways}-way")
    return fig

def plot_overview(jsonpath, rgb):
    """

    :param jsonpath: path to json file describing th task
    :param rgb: a numpy RGB array representation of the overall scene for plotting
    :return: figure
    """
    with open(jsonpath, "r") as f:
        data = json.load(f)

    radius_within_class = data["radius_within_class"]
    imagesize = data["imagesize"]
    scene_width = data["scene_width"]
    scene_height = data["scene_height"]
    image = data["image"]
    seed = data["seed"]
    num_shots = data["num_shots"]
    num_ways = data["num_ways"]

    fig, ax = plt.subplots()
    ax.imshow(rgb, origin='lower')
    ax.set_xlim(0, scene_width)
    ax.set_ylim(0, scene_height)
    for d in data["point_dict_info"]:
        ax.plot(d["x_proto"], d["y_proto"])
        ax.add_patch(plt.Circle((d["x_proto"], d["y_proto"]), radius_within_class, facecolor="none", edgecolor="k"))
        ax.text(d["x_proto"], d["y_proto"], d["class"], ha="center", va="center", fontsize=12)

        for k, xy in d["samples"].items():
            ax.add_patch(plt.Rectangle((xy["x"] - imagesize//2, xy["y"]-imagesize//2),
                                       width=imagesize, height=imagesize, facecolor="none", edgecolor="k"))
            ax.text(xy["x"], xy["y"], f"{d['class']}-{k}", ha="center", va="center", fontsize=6)

    ax.axis("off")
    ax.set_title(f"{image}-{seed} {num_shots}-shot {num_ways}-way")
    return fig

def sample_class_prototypes(num_class_prototypes, imagesize, scene_width, scene_height, radius_between_classes, invalid_mask):

    class_prototype_xy_list = []
    i = 0
    while(len(class_prototype_xy_list) < num_class_prototypes):
        i += 1
        if i > num_class_prototypes*20:
            raise ValueError(f"failed to find class prototypes after {i} tries maybe "
                             f"radius_between_classes={radius_between_classes} or "
                             f"too many num_class_prototypes={num_class_prototypes} is too large for "
                             f"an image of width={scene_width} height={scene_height}")

        # sample class prototype location
        y = np.random.randint(imagesize, scene_height-imagesize)
        x = np.random.randint(imagesize, scene_width-imagesize)

        # calculate euclidean distances to previous points
        distances = [np.linalg.norm(np.array([x, y]) - np.array([x_, y_])) for x_, y_ in class_prototype_xy_list]

        # reject point if too close to other class prototypes
        if any([d < radius_between_classes for d in distances]):
            continue

        # reject point if any pixel within imagesize is invalid
        if invalid_mask[y-imagesize//2:y+imagesize//2,x-imagesize//2:x+imagesize//2].any():
            continue

        # check if image is still fully in bound
        upper_bound = y - imagesize // 2 > 0
        lower_bound = y + imagesize // 2 < scene_height
        left_bound = x - imagesize // 2 > 0
        right_bound = x + imagesize // 2 < scene_width
        if not all([upper_bound, lower_bound, left_bound, right_bound]):
            continue


        class_prototype_xy_list.append((x, y))

    return class_prototype_xy_list

def sample_instances_for_prototype(num_instances, x_proto, y_proto, imagesize, invalid_mask, radius_within_class):
    scale = radius_within_class / 3  # 3 sigma radius covering 99.7% of sampled points

    scene_height, scene_width = invalid_mask.shape

    samples_xy_list = []
    i = 0
    while len(samples_xy_list) < num_instances:
        i += 1
        if i > num_instances*40:
            raise ValueError(f"failed to find enough samples for one class prototype after {i} tries. Maybe "
                             f"adjust num_instances")

        x,y = np.random.normal(loc=np.array([x_proto, y_proto]), scale=scale)
        x = int(x)
        y = int(y)

        # calculate euclidean distances to previous points
        distances = [np.linalg.norm(np.array([x, y]) - np.array([x_, y_])) for x_, y_ in samples_xy_list]

        # reject samples that are too close (images overlapping)
        if any([d < imagesize for d in distances]):
            continue

        # reject point if any pixel within imagesize is invalid
        if invalid_mask[y-imagesize//2:y+imagesize//2,x-imagesize//2:x+imagesize//2].any():
            continue

        # check if image is still fully in bound
        upper_bound = y - imagesize // 2 > 0
        lower_bound = y + imagesize // 2 < scene_height
        left_bound = x - imagesize // 2 > 0
        right_bound = x + imagesize // 2 < scene_width
        if not all([upper_bound, lower_bound, left_bound, right_bound]):
            continue

        samples_xy_list.append((x, y))

    return samples_xy_list


if __name__ == '__main__':
    main()
