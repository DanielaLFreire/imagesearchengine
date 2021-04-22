import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))#'.stem' take out the file extension 
features = np.array(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Delete previus uploaded images
        for root, dirs, files in os.walk("./static/uploaded/"):
            for file_img in files:
                os.remove(os.path.join(root, file_img))

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        per_sim = ((1-dists)*100).astype(int)
        per_sim = per_sim.astype(str)
              
        ids = np.argsort(dists)[:9]  # Top 9 results

        scores =[(img_paths[id].name+"\n"+"("+per_sim[id]+"%)", img_paths[id]) for id in ids] #iterante each element of array

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores
)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
