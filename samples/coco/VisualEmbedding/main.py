from images import Image2vec
from info import BBOX_FILE
import json


def main():
    first_run()


def first_run():
    img2id = dict()  # name image -> id image
    with open(BBOX_FILE, 'r') as bfile:
        bjs = json.loads(bfile.read())
        img2id = {image['file_name']: image['id'] for image in bjs['images']}
    img2vec = Image2vec(has_model=True, img2id=img2id)
    img2vec.compute_all_feats_and_store()


if __name__ == '__main__':
    main()
