from images import Image2vec
from info import BBOX_FILE, TRAIN_FILE, DUMP_DIR, IMG_DATA
import json
import os
#from gluoncv import data, utils
from matplotlib import pyplot as plt
import ast
import pickle


def main():
    # with open(TRAIN_FILE, "r") as f:
    #     for line in f.readlines():
    #         img_in, img_out, w_in, w_out, bbox_in, bbox_out = [ast.literal_eval(e) for e in line.split('\t')]
    #         width = bbox_in[2]
    #         height = bbox_in[3]
    #         if width > 224 or height > 224:
    #             print(img_in, width, height)
    # train_dataset = data.COCODetection(root='../DATA/', splits=['instances_train2014'])
    # train_image, train_label = train_dataset[0]
    # bounding_boxes = train_label[:, :4]
    # class_ids = train_label[:, 4:5]
    # print('Image size (height, width, RGB):', train_image.shape)
    # print('Num of objects:', bounding_boxes.shape[0])
    # print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
    #       bounding_boxes)
    # print('Class IDs (num_boxes, ):\n', class_ids)

    # utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
    #                     labels=class_ids, class_names=train_dataset.classes)
    # bbox_xs = [round(b[0]) for b in bounding_boxes]
    # bbox_ys = [round(b[1]) for b in bounding_boxes]
    # plt.scatter(bbox_xs, bbox_ys, color='red', s=40)
    # plt.show()
    #for i in range(2):
    #    print(train_dataset[i][0])
    first_run()
    test_first_run()


def first_run():
    with open(DUMP_DIR + "/word2id.pkl", "rb") as f:
        word2id = pickle.load(f)
    img2id = dict()  # name image -> id image
    with open(BBOX_FILE, 'r') as bfile:
        bjs = json.loads(bfile.read())
        img2id = {image['file_name']: image['id'] for image in bjs['images']}
        categories = {cat['id']: cat['name'] for cat in bjs['categories']}
    catid2wordid = dict()
    for ci in categories:
        if not (categories[ci] in word2id):
            if ''.join(categories[ci].split()) in word2id:
                catid2wordid[ci] = word2id[''.join(categories[ci].split())]
            elif categories[ci].split()[1] in word2id:
                catid2wordid[ci] = word2id[categories[ci].split()[1]]
        else:
            catid2wordid[ci] = word2id[categories[ci]]
    img2vec = Image2vec(has_model=True, img2id=img2id, catid2wordid=catid2wordid, with_bbox=True)
    img2vec.compute_all_feats_and_store()

def test_first_run():
    with open(DUMP_DIR + "/word2id.pkl", "rb") as f:
        word2id = pickle.load(f)
    img2vec = Image2vec(has_model=False, with_bbox=True)


if __name__ == '__main__':
    main()
