"""given a folder of images, measure the similarity of them with a given text description"""
import argparse
import glob
import open_clip
from open_clip import tokenizer
from PIL import Image
import numpy as np
import torch
import os
    
parser = argparse.ArgumentParser()
parser.add_argument('--input-files', type=str, required=True, help='input images pattern')
parser.add_argument('--description', type=str, required=True, help='input images pattern')
parser.add_argument('--threshold', type=float, default = 0.3, help='only show above threshold images')

args = parser.parse_args()

# model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model, _, preprocess = open_clip.create_model_and_transforms('mobilenetv2_050', pretrained='logs/mobilenetv2_050/checkpoints/epoch_199.pt')
text_tokens = tokenizer.tokenize([args.description])
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)


files = glob.glob(args.input_files)


scores = []
batch_size = 4
for filenames in np.array_split(files, len(files) // batch_size):
    images = []
    for filename in filenames:
        image = Image.open(filename).convert("RGB")
        images.append(preprocess(image))
    image_input = torch.tensor(np.stack(images))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = image_features.cpu().numpy() @ text_features.cpu().numpy().T
        print(similarity.shape)
        scores.append(similarity)

scores = np.concatenate(scores)
image_score = list(zip([os.path.basename(f) for f in files], scores[:, 0]))
image_score = [s for s in image_score if s[1] > args.threshold]

for i, s in image_score:
    print(f"{i}: {s}")
