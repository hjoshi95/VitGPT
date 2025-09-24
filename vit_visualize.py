from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json


def load_model_and_extractor(model_name="google/vit-base-patch16-224"):
    model = ViTModel.from_pretrained(model_name, output_attentions=True)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, feature_extractor, device

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

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

def plot_attention_overlay(blended_image, title_text=None, save_path=None):
    plt.figure(figsize=(6,6))
    plt.imshow(blended_image)
    plt.axis('off')
    if title_text is None:
        title_text = "ViT CLS Token Attention Heatmap"
    plt.title(title_text)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # Read image path and caption produced by vit_gpt2.py
    image_path = None
    caption = None
    try:
        with open("last_caption.json", "r") as f:
            data = json.load(f)
            image_path = data.get("image_path")
            caption = data.get("caption")
    except Exception:
        pass

    if image_path is None:
        image_path = "inputs/cat_park.jpg"
    if caption is None:
        caption = "ViT CLS Token Attention Heatmap"

    model, feature_extractor, device = load_model_and_extractor()
    image = load_image(image_path)
    attn_map = get_attention_map(model, feature_extractor, device, image)
    attn_normalized = smooth_and_resize_attention(attn_map, image.size)
    blended = overlay_attention_on_image(image, attn_normalized)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("attention_maps_visualized")
    out_path = os.path.join(out_dir, f"{base_name}_heatmap_captioned.png")
    plot_attention_overlay(blended, title_text=caption, save_path=out_path)
    print("Image:", image_path)
    print("Caption:", caption)
    print("Saved plotted heatmap with caption title to:", out_path)
