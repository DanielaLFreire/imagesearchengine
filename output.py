import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import pandas as pd

# Read image features

fe = FeatureExtractor()
features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))  # '.stem' take out the file extension
features = np.array(features)

array_output_pdf = []
array_output_jpg = []

for root, dirs, files in os.walk("./static/uploaded/"):
    for file_img in files:
        img = Image.open(root + file_img)  # PIL image

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
        per_sim = ((1 - dists) * 100).astype(int)
        per_sim = per_sim.astype(str)

        ids = np.argsort(dists)[:10]  # Top 10 results
        
        #Change the file names
        file_img_pdf = file_img.replace('.jpg', '.pdf')
        #Restore white spaces
        file_img_pdf = file_img_pdf.replace('_', ' ')

        for id in ids:
            #vetor of similar ('.jpg')
            array_output_jpg.append([file_img, img_paths[id].name, str(per_sim[id])])
		
            #Change the file names
            similar_name = img_paths[id].name.replace('.jpg', '.pdf')
            #Restore white spaces
            similar_name = similar_name.replace('_', ' ')
            #vetor of similar ('.pdf')
            array_output_pdf.append([file_img_pdf, similar_name, str(per_sim[id])])

    # dataframe
    df_output_jpg = pd.DataFrame(array_output_jpg)
    df_output_pdf = pd.DataFrame(array_output_pdf)

    # save csv
    df_output_jpg.to_csv("./static/output/" + 'output_jpg.csv', index=False, header=['ORIGINAL_IMG', 'SIMILAR', 'SIMILAR_PERC'])
    df_output_pdf.to_csv("./static/output/" + 'output_pdf.csv', index=False, header=['ORIGINAL_IMG', 'SIMILAR_IMG', 'SIMILAR_PERC'])
