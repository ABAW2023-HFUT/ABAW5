from torch.utils.data import Dataset
from utils import get_annotations, get_disregard, get_num_label
import os
import pandas as pd
import torch
import numpy as np
import cv2
import math

class PrepareDataset(Dataset):
    def __init__(self, root, transforms):
        super(PrepareDataset, self).__init__()
        self.transforms = transforms
        self.image_path = os.path.join(root,'aligned')
        self.video_list = []
        self.image_list = []
        for i in os.listdir(self.image_path):
            for j in os.listdir(os.path.join(self.image_path, i)):
                if j.split('.')[-1] != 'jpg':
                    continue
                self.video_list.append(i)
                self.image_list.append(j)
        self.length = len(self.image_list)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.video_list[idx], self.image_list[idx]))
        img = img[:, :, ::-1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.video_list[idx], self.image_list[idx]


class AffWild2_Dataset(Dataset):
    def __init__(self, args, phase):
        super(AffWild2_Dataset, self).__init__()
        annotations = get_annotations(args.task)
        root = args.root
        seq_len = args.seq_len
        seq_step = args.seq_step if phase == 'train' else args.seq_len
        phase = 'Train_Set' if phase == 'train' else 'Validation_Set'
        feat_path = [os.path.join(root,'features',x+'.npy') for x in args.feature]
        feat_list = [np.load(x, allow_pickle=True).item() for x in feat_path]
        label_path = os.path.join(root,'Annotations',annotations,phase)
        disregard = get_disregard(args.task)
        num_label = get_num_label(args.task)
        self.seq_list = []
        self.feat_dim = []
        for i in os.listdir(label_path):
            df = pd.read_table(os.path.join(label_path, i), header=None, sep=',')
            labels = np.array(df.iloc[1:,:num_label], dtype=np.float32)
            if args.task == 'EXPR':
                labels = labels.astype(int)
            anno_index = np.argwhere(np.sum(labels == disregard, axis=1) == 0).flatten()
            feat_index = np.array([int(x[:-4]) - 1 for x in feat_list[0][i[:-4]].keys()])
            index = sorted(set(anno_index).intersection(set(feat_index)))
            if self.feat_dim == []:
                self.feat_dim = [x[i[:-4]]['{}.jpg'.format(str(index[0] + 1).zfill(5))].shape[0] for x in feat_list]
            num_idx = len(index)
            num_seq = math.ceil(num_idx / seq_step)
            for idx in range(num_seq):
                seq_idx = index[idx * seq_step: min(idx * seq_step + seq_len, num_idx)]
                seq_feat = np.concatenate([np.array([y[i[:-4]]['{}.jpg'.format(str(x + 1).zfill(5))] for x in seq_idx]) for y in feat_list],axis=1)
                seq_label = labels[seq_idx,:]
                if seq_feat.shape[0] < seq_len:
                    seq_feat = np.pad(seq_feat, ((0, seq_len - seq_feat.shape[0]), (0, 0)), 'edge')
                    seq_label = np.pad(seq_label, ((0, seq_len - seq_label.shape[0]), (0, 0)), 'edge')
                self.seq_list.append({'feat':seq_feat, 'label':seq_label})
        self.length = len(self.seq_list)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.seq_list[idx]['feat'].astype(np.float32)), torch.from_numpy(self.seq_list[idx]['label']).squeeze()
    
    def get_feature_dim(self):
        return self.feat_dim

class Test_Dataset(Dataset):
    def __init__(self, args):
        super(Test_Dataset, self).__init__()
        root = args.root
        seq_len = args.seq_len
        feat_path = [os.path.join(root,'features',x+'.npy') for x in args.feature]
        feat_list = [np.load(x, allow_pickle=True).item() for x in feat_path]
        video_list = list(pd.read_table('./predict/{}.txt'.format(args.task), header=None).iloc[:,0])
        self.seq_list = []
        self.feat_dim = []
        for i in video_list:
            imgid = np.array(sorted([x for x in feat_list[0][i].keys()]))
            if self.feat_dim == []:
                self.feat_dim = [x[i][imgid[0]].shape[0] for x in feat_list]
            num_idx = len(imgid)
            num_seq = math.ceil(num_idx / seq_len)
            for idx in range(num_seq):
                seq_id = imgid[idx * seq_len: min((idx + 1) * seq_len, num_idx)]
                seq_length = len(seq_id)
                seq_feat = np.concatenate([np.array([y[i][x] for x in seq_id]) for y in feat_list],axis=1)
                if seq_feat.shape[0] < seq_len:
                    seq_feat = np.pad(seq_feat, ((0, seq_len - seq_feat.shape[0]), (0, 0)), 'edge')
                    seq_id = np.pad(seq_id, ((0, seq_len - seq_id.shape[0])), 'edge')
                self.seq_list.append({'feat':seq_feat, 'video':str(i), 'id':list(seq_id), 'length':seq_length})
        self.length = len(self.seq_list)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return self.seq_list[idx]['feat'].astype(np.float32), self.seq_list[idx]['video'], \
               self.seq_list[idx]['id'], self.seq_list[idx]['length']
    
    def get_feature_dim(self):
        return self.feat_dim
