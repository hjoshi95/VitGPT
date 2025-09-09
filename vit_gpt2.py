from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
import torch
import warnings
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
    if image_path.startswith("http") or image_path.startswith("https"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
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
    image_path = "inputs/man_bar.jpg"
    caption = generate_caption(image_path)
    print("Generated Caption:", caption) 
