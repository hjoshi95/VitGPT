from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import warnings
import json
import os
import argparse
warnings.filterwarnings("ignore", message=".*attention_mask.*")

model_name = "nlpconnect/vit-gpt2-image-captioning"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix the tokenizer padding token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(
        pixel_values, 
        max_length=16, 
        num_beams=4,
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        early_stopping=True
    )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for a local image and persist context for visualization")
    parser.add_argument("--image", required=True, help="Path to a local image file (e.g., inputs/cat_park.jpg)")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    caption = generate_caption(image_path)
    print("Generated Caption:", caption)
    payload = {"image_path": image_path, "caption": caption}
    with open("last_caption.json", "w") as f:
        json.dump(payload, f)
    print("Wrote last_caption.json with image path and caption.")
