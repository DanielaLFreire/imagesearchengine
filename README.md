# imagesearchengine
Image search engine using Python for bank payment slips of the TJSP

## Overview
- Simple image-based image search engine using Keras + Flask. You can launch the search engine just by running two python scripts.
- `offline.py`: This script extracts a deep-feature from each database image. Each feature is a 4096D fc6 activation from a VGG16 model with ImageNet pre-trained weights.
- `server.py`: This script runs a web-server. You can send your query image to the server via a Flask web-interface. The server finds similar images to the query by a simple linear scan.
- GPUs are not required.
- Tested on Ubuntu 18.04 and WSL2 (Ubuntu 20.04)
- OS: [e.g. Ubuntu 20.04. Note that this repo works on WSL, but may not work on a native Windows.]
- Python version: [e.g. Python 3.7.6]
- TensorFlow version: [e.g. TensorFlow 2.2.0. You can check it by python -c 'import tensorflow as tf; print(tf.__version__)']



## Usage
```bash
git clone https://github.com/DanielaLFreire/imagesearchengine
cd sis
pip install -r requirements.txt

# Then fc6 features are extracted and saved on static/feature
# Note that it takes time for the first time because Keras downloads the VGG weights.
python offline.py

# Now you can do the search via localhost:5000
python server.py
```
