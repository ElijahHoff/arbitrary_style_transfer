# 

AdaIN Style Transfer Project
I would like to present my personal implementation of arbitrary style transfer using the AdaIN (Adaptive Instance Normalization) technique. Although English is not my first language, I have done my best to make this documentation clear and grammatically correct.

Overview
This repository demonstrates a neural network that takes:

A content image (for the main structure and shapes).
A style image (for the artistic look and textures).
And outputs a stylized image that merges the content’s structure with the style’s color and brushstroke patterns. The approach is based on the paper “Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization”.

Repository Structure
bash
Copy
my_style_transfer_project/
  models/
    vgg_encoder.py         # Pretrained VGG19 feature extractor
    decoder.py             # Network to reconstruct image from features
    style_transfer_net.py  # Wrapper combining Encoder, AdaIN, and Decoder
  weights/
    best_decoder.pth       # Trained model weights
  streamlit_app.py         # Streamlit-based web interface for demo
  requirements.txt         # Python dependencies
  README.md                # You are here
Key Components
VGGEncoder (in vgg_encoder.py): Extracts high-level features from an input image using parts of VGG19.
Decoder (in decoder.py): Converts those feature maps back into a 3-channel image.
StyleTransferNet (in style_transfer_net.py): Implements the AdaIN logic to adapt content features to the style’s statistics, followed by decoding.
Installation
Clone this repository:
bash
Copy
git clone https://github.com/ElijahHoff/arbitrary_style_transfer.git
cd my_style_transfer_project
Install required packages:
bash
Copy
pip install -r requirements.txt
This should install PyTorch, TorchVision, Pillow, Streamlit, and any other libraries needed.

There is a feature of how stong impact of style is.

Training
If you wish to re-train the model (for instance on a smaller subset of COCO and WikiArt), you can create a separate script, such as train.py, that:



Loads and freezes the VGG encoder.
Initializes the decoder for training.
Uses AdaIN-based losses (content + style).
Saves the best weights to best_decoder.pth.
(Note: The training procedure might require substantial GPU resources. I used Google Colab A100 for my experiments.)

Inference
To perform style transfer on your own images:

Make sure best_decoder.pth is inside weights/.
Write or adapt a script (for example infer.py) that:
bash
Copy
python infer.py --content path/to/content.jpg --style path/to/style.jpg --output result.jpg
The script loads VGGEncoder and Decoder, composes StyleTransferNet, and applies AdaIN to produce the stylized image.
The output image is saved in the specified path (result.jpg).
Streamlit Demo
A simple interactive interface is available via Streamlit:

Run:
bash
Copy
streamlit run streamlit_app.py
This will launch a local web application at http://localhost:8501.
Upload one image as content and another as style, then adjust the alpha slider if desired, and click Transfer Style to see the resulting stylized image.
Future Work
Larger Datasets: Training on many style samples (e.g., WikiArt) improves the variety of styles.
Higher Resolution: You can experiment with bigger image sizes if hardware allows.
Advanced Techniques: Consider other normalization approaches, skip connections, or multi-layer style mixing.
Acknowledgements
The official paper about AdaIN:
X. Huang and S. Belongie, “Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization” (2017).

PyTorch and TorchVision developers for providing user-friendly deep learning libraries.

License
The code in this repository is provided under the MIT License. Feel free to modify, distribute, or adapt it for your purposes. If you use it in academic work, a citation or mention would be appreciated.
