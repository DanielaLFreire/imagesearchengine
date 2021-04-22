import PIL.Image as Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
Image.MAX_IMAGE_PIXELS = None

#img_path = "./static/img/"
img_path = "./static/aug/"

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path(img_path).glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
