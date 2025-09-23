# VitGPT
# VitGPT : Image captioning using Vision Transformer (ViT) by visualization attention heatmaps.




In this little project, I try to discover the power of attention models for both images and text.
I wanted to understand how will a caption be generated for an image. In the past, I have generated captions on images on FlickR images using LSTMs.
However, LSTMs cannot capture long range dependencies, and encoder CNNS comprise of locality biases due to their kernels.
By combining a Encoder Vision Transformer Model with a Decoder Transformer Model (GPT2) - I wanted to see how this limitation is addressed since Transformers do a wonderful job capturing long dependencies using Self Attention. 
A Vision Transformer leverages the Attention mechanism where an image is also prefaced to have the ability to learn context by breaking it down into image patch grids similar to tokens in a text. 


I have basically run some inference here where I've use pre-trained models.
I get the attention weights, normalize them and apply a scaled heatmap on top of the image for a generated caption. This basically visualizes the parts of the image that were attended to when the caption was generated


**avg_attn = last_layer_attn.mean(dim=0):**
--> The attention weights from a single layer have a shape of (num_heads, num_patches + 1, num_patches + 1), where num_heads is the number of attention heads. To get a single, coherent attention map, the weights are averaged across all the attention heads.

**cls_attn = avg_attn[0, 1:]:**
-->  In ViT models, a special class token is added to the sequence of image patches. This token's attention to the other patches is often used to represent the model's overall focus. The [0, 1:] index selects the attention weights from the class token (0) to all the image patches (1:), effectively showing which parts of the image the model focused on to make its classification or prediction.

**attn_map = cls_attn.reshape(14, 14).cpu().numpy():**
-->This reshapes the 1D attention scores into a 2D map. For a 224x224 pixel image with a patch size of 16x16, there are (224/16) * (224/16) = 14 * 14 = 196 patches. The attention scores are reshaped into a 14x14 grid, which represents the spatial layout of the original image.


## Results 
### Generated Caption: a gray and white cat laying in the grass 
<img width="600" height="600" alt="Attention_maps" src="https://github.com/user-attachments/assets/74942069-fa54-4a69-9d1d-4a415ae938c2" />

### Generated Caption: a man sitting at a bar looking at his cell phone 
<img width="600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/7dbba049-0820-4ab7-a4e7-a22d60a10df4" />


## Running 
1. Add an image to the input folder
2. Change the path in the image_url variable to _input/path_to_your_img.jpeg_
3. Run python vit_gpt2.py to generate a caption for the input image
4. Run python vit_visualize.py to visualize the attention weights on the input image

## Next - I plan on Fine-Tuning the FlickR dataset using these transformer models 
