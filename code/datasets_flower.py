from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import json

import nltk, sklearn

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def prepare_data(data):
    # imgs, captions, captions_lens, class_ids, keys = data

    imgs, captions, captions_lens, class_ids, keys, wrong_caps, \
        wrong_caps_len, wrong_cls_id, noise, word_labels = data

    # imgs, captions, captions_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id, noise, word_labels = data

    # imgs, caps, cap_len, imgs2, caps2, cap_len2, cls_id, key  = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    noise = noise[sorted_cap_indices]
    word_labels = word_labels[sorted_cap_indices]

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    w_sorted_cap_lens, w_sorted_cap_indices = \
        torch.sort(wrong_caps_len, 0, True)

    wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
    wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

    if cfg.CUDA:
        wrong_caps = Variable(wrong_caps).cuda()
        w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
    else:
        wrong_caps = Variable(wrong_caps)
        w_sorted_cap_lens = Variable(w_sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, wrong_caps, w_sorted_cap_lens, wrong_cls_id, noise, word_labels]


def get_imgs(img_path, imsize, flip, x, y, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
        ## crop
        img = img.crop([x, y, x + 256, y + 256])
        if flip:
            img = F.hflip(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i == 0 or i == 1:
                continue
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.dict_name2cap = {}

        self.class_id = self.load_class_id(split_dir, len(self.filenames['img']))
        self.number_example = len(self.filenames['img'])

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames['img'])):
            cap_path = '%s/%s.txt' % ('E:\\anhang\\Lightweight-Manipulation-master\\data\\flower\\text', filenames['img'][i])
            # cap_path =  +filenames['img'][i]
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        cap = 'this flower'
                        cap = cap.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cap.lower())

                        print('cap', cap)
                        # continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames['img'][i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        # train_names = self.load_filenames(data_dir, 'train')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        with open('E:\\anhang\\Lightweight-Manipulation-master\\data\\cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        dic_class = []
        dic_classs = {}
        for key, value in cat_to_name.items():
            dic_class.append(value)
        for i in range(len(dic_class)):
            dic_classs[dic_class[i]] = i

        return dic_classs

    def load_filenames(self, data_dir, split):
        # filepath = 'E:\\anhang\\Lightweight-Manipulation-master\\data\\flower_cat_dic.pkl'
        filepath = 'E:\\anhang\Lightweight-Manipulation-master\data\\flower_cat_dic.pkl'
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames['img'])))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames['img'][index]
        cat = self.filenames['cat'][index]
        cls_id = self.class_id[cat]
        # cls_id = self.class_id[index]
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(nz).normal_(0, 1))

        #
        bbox = None

        flip = random.rand() > 0.5
        new_w = new_h = int(256 * 76 / 64)
        x = random.randint(0, np.maximum(0, new_w - 256))
        y = random.randint(0, np.maximum(0, new_h - 256))
        img_name = '%s/jpg/%s.jpg' % (self.data_dir, key)
        imgs = get_imgs(img_name, self.imsize, flip, x, y,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        wrong_idx = random.randint(0, len(self.filenames['img']))
        wrong_new_sent_ix = wrong_idx * self.embeddings_num + sent_ix
        wrong_caps, wrong_cap_len = self.get_caption(wrong_new_sent_ix)
        cat_index = self.filenames['cat'][wrong_idx]
        wrong_cls_id = self.class_id[cat_index]

        caption = ""
        for i in caps:
            if i != 0:
                caption += self.ixtoword.get(int(i[0])) + " "

        list_sents = nltk.tokenize.sent_tokenize(text=caption)
        list_tokens = nltk.tokenize.word_tokenize(text=list_sents[0])
        list_pos = nltk.tag.pos_tag(list_tokens)

        new_len = 0
        word_labels = []
        for i in list_pos:
            new_len += 1
            if "NN" in i[1] or "JJ" in i[1]:
                word_labels.append(np.array(1))
            else:
                word_labels.append(np.array(0))

        if new_len < cfg.TEXT.WORDS_NUM:
            for i in range(0, cfg.TEXT.WORDS_NUM - new_len):
                word_labels.append(np.array(0))
        else:
            word_labels = word_labels[:cfg.TEXT.WORDS_NUM]

        word_labels = np.asarray(word_labels)

        return imgs, caps, cap_len, cls_id, key, wrong_caps, \
            wrong_cap_len, wrong_cls_id, noise, word_labels

    def __len__(self):
        return len(self.filenames['img'])