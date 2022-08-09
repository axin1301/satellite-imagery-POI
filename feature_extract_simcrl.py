import math
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from dataloader import *
from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
import torch
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES']='7'
import setproctitle
setproctitle.setproctitle('setproctitle')

def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x,y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    print(train_X.shape)
    
    #test_X, test_y = inference(test_loader, simclr_model, device)
    #print(test_X.shape)
    return train_X, train_y#, test_X, test_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset_lr(
        './img_name_all.csv',#beijing satellite image
        './POI_array_Sat_cate_1st.txt', #POI array, it is a null variable, not used here
        './Shanghai/BJ_zl15_unified/',
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )

    print('train_set',len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)

    model_fp = os.path.join('checkpoint_file/', "checkpoint_100.tar")
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y) = get_features(
        simclr_model, train_loader, args.device
    )
    
    np.savetxt('data_feature_BJ.txt',(train_X),fmt='%f') # save the image embeddings

