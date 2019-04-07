import os
from os import listdir
import numpy as np
from images.deep_learning_models import VGG16
from keras.models import Model
from tqdm import tqdm
from info import IMG_DATA, IMG_FEATS, BBOX_FILE
import pickle
import json
import tensorflow as tf
import math
from matplotlib import pyplot as plt
#tf.enable_eager_execution() # Enable this if you want to visualize images. Remember to remove VGG


class Image2vec(object):
    """Image2vec"""
    def __init__(self, has_model=False, img2id=None, catid2wordid=None, with_bbox=False):
        self.image_feats = np.array([])
        self.ids = None
        self.img2id = img2id
        self.with_bbox = with_bbox
        self.catid2wordid = catid2wordid
        if os.path.exists(IMG_FEATS) and has_model == False:
            self._load_with_pickle()
        if not os.path.exists(IMG_FEATS) or has_model:
            vgg_full_model = VGG16(weights='imagenet')
            #We don't need the prediction layer
            self.vgg = Model(input=vgg_full_model.input, output=vgg_full_model.get_layer('block5_pool').output)
            self.session = tf.Session()
        if not os.path.exists(IMG_FEATS):
            os.makedirs(IMG_FEATS)

    def get_features(self, img_id, bjs):
        feats = self._lookup(img_id)
        if feats is not None:
            return feats
        return self._compute_features(img_id) if not self.with_bbox else self._compute_features_with_bbox(img_id, bjs)

    def _compute_features(self, img_id):
        from keras.preprocessing import image
        from images.deep_learning_models.imagenet_utils import preprocess_input
        img = image.load_img(IMG_DATA + "/" + img_id, target_size=(224, 224)) #[224 x 224]
        img_array = image.img_to_array(img) #[224 x 224 x channels]
        print(img_array.shape)
        img_array = np.expand_dims(img_array, axis=0) #[1 x 224 x 224 x channels]
        #Subtract the mean RGB channels of the imagenet dataset
        #(since the model has been trained on a different dataset)
        img_array = preprocess_input(img_array) #[1 x 224 x 224 x channels]
        feats = self.vgg.predict(img_array) #[1 x 7 x 7 x 512]
        feats = np.reshape(feats, [1, 49, 512]) #[1 x 49 x 512]
        if self.image_feats.ndim == 1:
            self.image_feats = np.array(feats)
            self.ids = np.array(self.img2id[img_id])
        else:
            self.image_feats = np.vstack((self.image_feats, feats))
            self.ids = np.vstack((self.ids, [self.img2id[img_id]]))
        #self._save(img_id, feats[0])
        return feats[0]

    def _compute_features_with_bbox(self, img_id, bjs):
        from keras.preprocessing import image
        from images.deep_learning_models.imagenet_utils import preprocess_input
        img = image.load_img(IMG_DATA + "/" + img_id) #[height x width]
        img_array = image.img_to_array(img) #[height x width x channels]
        #plt.imshow(img_array/255)
        #plt.savefig("fig_before.jpg")
        anns = [a for a in bjs['annotations'] if a['image_id'] == self.img2id[img_id]]
        for a in anns:
            bbox = [round(c) for c in a['bbox']]
            category = self.catid2wordid[a['category_id']]
            #img_cropped = tf.image.crop_to_bounding_box(img_array, bbox[1], bbox[0], bbox[3], bbox[2])
            img_cropped = img_array[bbox[1]:(bbox[1] + bbox[3] + 1), bbox[0]:(bbox[0] + bbox[2] + 1), :]
            height, width = img_cropped.shape[:-1]
            img_cropped = np.expand_dims(img_cropped, axis=0) #[1 x height x width x channels]
            #Subtract the mean RGB channels of the imagenet dataset
            #(since the model has been trained on a different dataset)
            #img_cropped = preprocess_input(img_cropped) #[1 x height x width x channels]
            image_ph = tf.placeholder(tf.float32, name="image_ph", shape=[1, None, None, 3])
            feed_dict = {image_ph: img_cropped}
            if height > 224 or width > 224:
                img_cropped = tf.compat.v1.image.resize_bilinear(image_ph, [min(224, height), min(224, width)])
            img_cropped = tf.compat.v1.image.resize_image_with_crop_or_pad(img_cropped, *[224, 224]).eval(feed_dict, session=self.session)
            #img_cropped = tf.keras.backend.resize_images(img_cropped, math.ceil(224 / int(height)), math.ceil(round(224 / int(width))), "channels_last", interpolation='nearest')
            #plt.imshow(img_cropped[0]/255)
            #plt.savefig("fig_after.jpg")
            feats = self.vgg.predict(img_cropped) #[1 x 7 x 7 x 512]
            feats = np.reshape(feats, [1, 49, 512]) #[1 x 49 x 512]
            if self.image_feats.ndim == 1:
                self.image_feats = np.array(feats)
                self.ids = np.array(str(self.img2id[img_id]) + "_" + str(category))
            else:
                self.image_feats = np.vstack((self.image_feats, feats))
                self.ids = np.vstack((self.ids, [str(self.img2id[img_id]) + "_" + str(category)]))
        return 0 #return feats[0]

    def compute_all_feats_and_store(self):
        print("Computing image features..")
        img_files = os.listdir(IMG_DATA)
        bar = tqdm(range(len(img_files)))
        n_split = 0
        with open(BBOX_FILE, 'r') as f:
            bjs = json.loads(f.read())
            for img_index in bar:
                img = img_files[img_index]
                if not (img.startswith(".") or os.path.isdir(img)) and img.endswith("jpg"):
                    self.get_features(img, bjs)
                    #self._compute_features(img)
                if img_index > 0 and img_index % 500 == 0:
                    print("Saving at index: ", img_index)
                    with open(IMG_FEATS + "/all_{}.pickle".format(n_split), "wb+") as f:
                        pickle.dump(self.image_feats, f)
                    with open(IMG_FEATS + "/ids_{}.pickle".format(n_split), "wb+") as f:
                        pickle.dump(self.ids, f)
                    n_split += 1
                    self.image_feats = np.array([])
                    self.ids = None
        with open(IMG_FEATS + "/all_{}.pickle".format(n_split), "wb+") as f:
            pickle.dump(self.image_feats, f)
        with open(IMG_FEATS + "/ids_{}.pickle".format(n_split), "wb+") as f:
            pickle.dump(self.ids, f)

    def _lookup(self, img_id):
        indexes = np.where(self.ids == str(self.img2id[img_id]))[0]
        return self.image_feats[indexes[0]] if indexes.size > 0 else None

    def _save(self, img, vectors):
        if not os.path.exists(IMG_FEATS):
            os.makedirs(IMG_FEATS)
        np.savetxt(IMG_FEATS + "/{}.txt".format(img[:-4]), vectors)

    def _load(self):
        print("Loading image features..")
        feat_files = os.listdir(IMG_FEATS)
        bar = tqdm(range(len(feat_files)))
        for feat_index in bar:
            feat_file = feat_files[feat_index]
            img_name = feat_file
            features = np.expand_dims(np.array(np.loadtxt(IMG_FEATS + "/" + feat_file)), axis=0)
            if self.image_feats.ndim == 1:
                self.image_feats = features
                self.ids = np.array(self.img2id[img_name])
            else:
                self.image_feats = np.vstack((self.image_feats, features))
                self.ids = np.vstack((self.ids, [self.img2id[img_name]]))

    def _load_with_pickle(self):
        with open(IMG_FEATS + "/all.pickle", "rb") as f:
            self.image_feats = pickle.load(f)
        with open(IMG_FEATS + "/ids.pickle", "rb") as f:
            self.ids = pickle.load(f).flatten()
        print(self.ids)
