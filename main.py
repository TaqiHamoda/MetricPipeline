import yaml, os, gc, tqdm

from src.parsers import Agisoft, Meshroom
from src.colormap import Colormap
from src.pipeline import Pipeline


if __name__ == "__main__":
    if not os.path.exists('config.yaml') or not os.path.isfile('config.yaml'):
        print("Couldn't find config file 'config.yaml'")
        exit()

    with open('config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    target_classes = config_data["target_classes"]

    masks_dir = config_data["masks_dir"]
    images_dir = config_data["images_dir"]
    depths_dir = config_data["depths_dir"]

    info_file = config_data["info_file"]
    camera_file = config_data["camera_file"]

    metric_file = config_data["metric_file"]

    colormap_path = config_data["colormap_path"]

    dirs = config_data["dirs"]

    colormap = Colormap(colormap_path)

    for d in tqdm.tqdm(dirs):
        imgs = []
        for img in os.listdir(f"{d}/{masks_dir}"):
            if "_overlay" in img:
                continue

            imgs.append(img.split('.')[:-1])

        sensor = Agisoft().parse(f"{d}/{camera_file}")[0] if ".xml" in camera_file else Meshroom().parse(f"{d}/{camera_file}")[0]

        pipeline = Pipeline(
            sensor = sensor,
            colormap=colormap,
            target_classes=target_classes
        )

        with open(f"{d}/{info_file}", "r") as f:
            info = yaml.safe_load(f)

            camera_type = info["camera_type"]
            visibility = info["visibility"]

        pipeline.processImages(
            imgs_path=[f"{d}/{images_dir}/{img}.jpg" for img in imgs],
            masks_path=[f"{d}/{masks_dir}/{img}.png" for img in imgs],
            depths_masks=[f"{d}/{depths_dir}/{img}.png" for img in imgs],
            camera_type=camera_type,
            visibility=visibility,
        )

        pipeline.toFile(f"{d}/{metric_file}")

        del pipeline
        gc.collect()
