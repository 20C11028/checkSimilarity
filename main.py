# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import clip
from PIL import Image
import numpy as np
import keras.backend as K
import tensorflow as tf

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def extractFeature(text, imageUrl, model):
    model, preprocess = clip.load(model, device='cuda')
    image = Image.open(imageUrl)
    image_input = preprocess(image).unsqueeze(0).cuda()
    text_input = clip.tokenize(text).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()

    return text_features, image_features

def Manhattan_distance(A,B):
   return K.sum( K.abs( A-B),axis=1,keepdims=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    text_features, image_features = extractFeature("a apple", "testPic/apple.png", "ViT-B/32")
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = (100 * image_features @ text_features.T)
    # print(similarity)
    # print(type(text_features))
    # print(image_features)
    # cosine
    cos = torch.nn.CosineSimilarity(dim=1)
    print("Cosine Similarity:", cos(text_features, image_features)[0])
    print(K.get_value(cos(text_features, image_features)[0]))

    text_features = text_features.cpu()
    image_features = image_features.cpu()
    print("Euclidean distance: ", np.linalg.norm(text_features - image_features))
    # print(tf.sqrt(tf.reduce_sum(tf.square(text_features - image_features), 1)))
    # print(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(text_features - image_features), 1))))

    print(K.get_value(Manhattan_distance(text_features, image_features)[0]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
