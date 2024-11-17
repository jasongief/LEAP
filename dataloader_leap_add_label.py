import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import pdb


categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
              'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
              'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
              'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
              'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
              'Clapping']


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y


def _get_temporal_visual_pseudo_label(frames_pseudo_label_names_list, expand_video_labels=False):
    if expand_video_labels:
        frames_pseudo_label_names_list = [frames_pseudo_label_names_list] * 10
    frames_temporal_pseudo_visual_labels = np.zeros((10, len(categories))).astype(np.float32) # [10, 25]
    for frame_id, frame_pseudo_label_name in enumerate(frames_pseudo_label_names_list):
        one_second_label = ids_to_multinomial(frame_pseudo_label_name)
        frames_temporal_pseudo_visual_labels[frame_id] = one_second_label
    return frames_temporal_pseudo_visual_labels


def _get_temporal_audio_pseudo_label(frames_pseudo_label_names_list, expand_video_labels=False):
    if expand_video_labels:
        frames_pseudo_label_names_list = [frames_pseudo_label_names_list] * 10
    frames_temporal_pseudo_audio_labels = np.zeros((10, len(categories))).astype(np.float32) # [10, 25]
    for frame_id, frame_pseudo_label_name in enumerate(frames_pseudo_label_names_list):
        one_second_label = ids_to_multinomial(frame_pseudo_label_name)
        frames_temporal_pseudo_audio_labels[frame_id] = one_second_label
    return frames_temporal_pseudo_audio_labels


class LLP_dataset(Dataset):
    def __init__(self, label, audio_dir, video_dir, st_dir,
                 transform=None, a_smooth=1.0, v_smooth=0.9,
                 v_pseudo_flag=True, a_pseudo_flag=True,
                 dataset_label_embedding_dir=None, word_embedding_dim=300):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform
        self.a_smooth = a_smooth
        self.v_smooth = v_smooth
        self.v_pseudo_flag = v_pseudo_flag
        self.a_pseudo_flag = a_pseudo_flag

        self.dataset_label_embedding_dir = dataset_label_embedding_dir
        self.word_embed_dim = word_embedding_dim


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        
        video_label_names = row[-1].split(',')
        video_label = ids_to_multinomial(video_label_names)

        #! use segment-level pseudo labels
        if self.v_pseudo_flag:
            base_path = "../data/CLIP/segment_pseudo_labels" # you may need to change this path 
            temporal_pseudo_visual_labels = np.load(os.path.join(base_path, name+'.npy')) # [10, 25]
            Pv = (np.sum(temporal_pseudo_visual_labels, axis=0) != 0).astype(np.float32)
        else:
            Pv = self.v_smooth * video_label + (1 - self.v_smooth) * 0.5
            temporal_pseudo_visual_labels = _get_temporal_visual_pseudo_label(video_label_names, expand_video_labels=True) # [10, 25]
            temporal_pseudo_visual_labels = self.v_smooth * temporal_pseudo_visual_labels + (1 - self.v_smooth) * 0.5


        if self.a_pseudo_flag:
            base_path = "../data/CLAP/segment_pseudo_labels" # you may need to change this path 
            temporal_pseudo_audio_labels = np.load(os.path.join(base_path, name+'.npy')) # [10, 25]
            Pa = (np.sum(temporal_pseudo_audio_labels, axis=0) != 0).astype(np.float32)
        else:
            Pa = self.a_smooth * video_label + (1 - self.a_smooth) * 0.5 # audio label is without label smoothing
            temporal_pseudo_audio_labels = _get_temporal_audio_pseudo_label(video_label_names, expand_video_labels=True) # [10, 25]


        #! add label embedding
        dataset_label_embed_ori = torch.load(self.dataset_label_embedding_dir) # type: dict
        dataset_label_embed = np.zeros((len(categories), self.word_embed_dim)).astype(np.float32) # [1, 25, 300/768]
        idx = 0
        for key, value in dataset_label_embed_ori.items():
            # print(key)
            dataset_label_embed[idx, :] = value.numpy()
            idx += 1
        assert idx == len(categories), 'the number of label embeddings is not equal to the length of category list'
        

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st,
                  'label': video_label, 'pa': Pa, 'temporal_pa': temporal_pseudo_audio_labels,
                  'pv': Pv, 'temporal_pv': temporal_pseudo_visual_labels,
                  'label_text_embed': dataset_label_embed,
                  'idx': np.array([idx]), "video_name": name}
        if self.transform:
            sample = self.transform(sample)

        return sample



class ToTensor:
    def __call__(self, sample):
        tensor = dict()
        for key in sample:
            if key != "video_name":
                tensor[key] = torch.from_numpy(sample[key])
            else:
                tensor["video_name"] = sample[key]
        return tensor