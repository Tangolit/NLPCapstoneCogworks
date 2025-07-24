import mygrad as mg
from mygrad.losses import margin_ranking_loss
from cogworks_data.language import get_data_path
from pathlib import Path
import json
import pickle
import random
from mygrad.nnet.initializers import glorot_normal
from mygrad import Tensor
import os
import torch

with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)
filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)

def assign_train_test():
    """
    Assigns train and test sets for the COCO dataset.
    Returns a dictionary with 'train' and 'test' keys.
    """
    train = {}
    test = {}
    counter = 0
    random.shuffle(coco_data["images"])
    for image in coco_data["images"]:
        if resnet18_features[image["id"]] is not None:
            counter+=1
            if counter % 5== 0:
                test.update({image["id"]: image["coco_url"]})
            else:
                train.update({image["id"]: image["coco_url"]})
    return {
        train, test    
    }

def get_triplets(batch_size):
    ##img_to_caps = dictionary created by Rian and his teammate to find valid caption ids
    #  # pulled from img_to_caps directly
    triplets = []
    batch_size = batch_size
    train, test = assign_train_test()
    image_ids = list(train.keys())
    random.shuffle(image_ids)
    for i in range(batch_size):
        img_id = image_ids[i]
        for caption in coco_data["annotations"]:
            if caption["image_id"] == img_id:
                true_caption = caption["caption"]
                break   
        triplets.append(true_caption, img_id, random.choice([i for i in image_ids if i != img_id]))
    return triplets

#Not finished
def compute_loss_accuracy(triplet, w_img):
    margin = 0.1
    cap, img_id_good, img_id_bad = triplet
    w_caption_bad = 
    sim_g = mg.dot(w_img, cap)
    sim_b = mg.dot(w_img, w_caption_bad)
    return mg.nnet.losses.margin_ranking_loss(sim_g, sim_b, margin=margin)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    torch.load(model.state_dict(), path)

class Img2Cap:
    def __init__(self, input_dim = 512, embed_dim=200):
        self.W = Tensor(glorot_normal((input_dim, embed_dim)), constant=False)
    
    def __call__(self, x):
        out = x @ self.W
        return out / mg.linalg.norm(out, axis=1, keepdims = True)
    
    def parameters(self):
        return [self.W]
    
#Not Finished
def train_model(mode, num_epochs, batch_size, learning_rate):
    optimizer = mg.nn.SGD(model.parameters(), learning_rate=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0

        triplets = get_triplets(batch_size)

        for triplet in triplets:

            caption, img_id_good, img_id_bad = triplet
            img_good = resne