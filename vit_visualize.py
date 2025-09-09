from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_model_and_extractor(model_name="google/vit-base-patch16-224"):
    model = ViTModel.from_pretrained(model_name, output_attentions=True)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, feature_extractor, device

def load_image(image_path_or_url):
    if image_path_or_url.startswith("http") or image_path_or_url.startswith("https"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    return image

def get_attention_map(model, feature_extractor, device, image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions
    last_layer_attn = attentions[-1][0]  # batch 0
    avg_attn = last_layer_attn.mean(dim=0)
    cls_attn = avg_attn[0, 1:]
    attn_map = cls_attn.reshape(14, 14).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map

def smooth_and_resize_attention(attn_map, image_size):
    attn_map_resized = cv2.resize(attn_map, image_size)
    attn_map_blur = cv2.GaussianBlur(attn_map_resized, (21,21), sigmaX=0, sigmaY=0)
    attn_normalized = attn_map_blur / attn_map_blur.max()
    return attn_normalized

def overlay_attention_on_image(image, attn_normalized, alpha=0.4):
    image_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return blend

def plot_attention_overlay(blended_image):
    plt.figure(figsize=(6,6))
    plt.imshow(blended_image)
    plt.axis('off')
    plt.title("ViT CLS Token Attention Heatmap")
    plt.show()

if __name__ == "__main__":
    model, feature_extractor, device = load_model_and_extractor()
    image_url = "man_bar.jpg"
    image = load_image(image_url)
    attn_map = get_attention_map(model, feature_extractor, device, image)
    attn_normalized = smooth_and_resize_attention(attn_map, image.size)
    blended = overlay_attention_on_image(image, attn_normalized)
    plot_attention_overlay(blended)
