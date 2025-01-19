import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from models.vgg_encoder import VGGEncoder
from models.decoder import Decoder
from models.style_transfer_net import StyleTransferNet
from models.utils import denormalize, adain


@st.cache_resource
def load_model():
    """
    Кэшируем загрузку модели, чтобы при повторном запуске не тратить время.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VGGEncoder().to(device).eval()
    
    decoder = Decoder().to(device)
    # Загружаем веса декодера
    decoder_weights_path = os.path.join("weights", "best_decoder.pth")
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
    decoder.eval()
    
    net = StyleTransferNet(encoder, decoder).to(device).eval()
    return net, device

def denormalize(tensor):
    """
    Обратная нормализация, если мы использовали mean=(0.485,0.456,0.406) и std=(0.229,0.224,0.225).
    """
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

###########################
# Основная часть приложения
###########################

st.title("Arbitrary Style Transfer (AdaIN)")

st.write("""
Upload a **content** image and a **style** image, then click "Transfer Style"
to generate a stylized image using the AdaIN approach.
""")

content_file = st.file_uploader("Upload Content Image", type=["jpg","jpeg","png"])
style_file   = st.file_uploader("Upload Style Image",   type=["jpg","jpeg","png"])

alpha = st.slider("Alpha (style strength)", 0.0, 1.0, 1.0, 0.1)

# Загрузка модели
net, device = load_model()

if st.button("Transfer Style"):
    if not content_file or not style_file:
        st.warning("Please upload both content and style images.")
    else:
        # Открываем как PIL
        content_pil = Image.open(content_file).convert("RGB")
        style_pil   = Image.open(style_file).convert("RGB")

        # Показать загруженные изображения
        st.image([content_pil, style_pil], caption=["Content", "Style"], width=300)

        # Трансформация
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        content_tensor = transform(content_pil).unsqueeze(0).to(device)
        style_tensor   = transform(style_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(content_tensor, style_tensor, alpha=alpha)

        # Денормализация
        out_img = denormalize(output.squeeze(0)).cpu()
        out_img = out_img.squeeze(0)
        # Перевод в PIL
        out_pil = transforms.ToPILImage()(out_img)

        # Отображаем результат
        st.subheader("Result")
        st.image(out_pil, caption="Stylized Image", use_container_width=True)
