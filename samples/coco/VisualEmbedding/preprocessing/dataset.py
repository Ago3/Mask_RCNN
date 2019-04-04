from random import shuffle, sample
import json
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pickle
from tqdm import tqdm
from os import path


# traffic light
# fire hydrant
# stop sign
# parking meter
# sports ball
# baseball bat
# baseball glove
# tennis racket
# wine glass
# hot dog
# potted plant
# dining table
# cell phone
# teddy bear
# hair drier

def compute_embeddings_and_dictionaries(glove_file, dump_dir=None):
    VOCAB_SIZE = 500000
    glove2word2vec(glove_input_file=glove_file, word2vec_output_file="gensim_glove_vectors.txt")
    from gensim.models.keyedvectors import KeyedVectors
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    embeddings = np.zeros((VOCAB_SIZE, 300))
    #words = np.chararray((VOCAB_SIZE), unicode=True)
    word2id, id2word = dict(), dict()
    word2id['unk'] = 0
    id2word[0] = 'unk'
    embeddings[0] = glove_model.wv['unk']
    unk_index = 171973
    for i in range(VOCAB_SIZE): #len(glove_model.wv.vocab)
        if i < unk_index: j = i + 1
        elif i == unk_index: continue
        else: j = i
        embedding_vector = glove_model.wv[glove_model.wv.index2word[i]]
        if embedding_vector is not None:
            embeddings[j] = embedding_vector
            #words[i] = glove_model.wv.index2word[i]
            word2id[glove_model.wv.index2word[i]] = j
            id2word[j] = word2id[glove_model.wv.index2word[i]]
    if dump_dir:
        print("Saving embedding files..")
        with open(dump_dir + "/word2id.pkl", 'wb+') as f:
            pickle.dump(word2id, f)
        with open(dump_dir + "/id2word.pkl", 'wb+') as f:
            pickle.dump(id2word, f)
        with open(dump_dir + "/embeddings.pkl", 'wb+') as f:
            pickle.dump(embeddings, f)
        print("done!")
    return word2id, id2word, embeddings

def load_embeddings_and_dictionaries(dump_dir):
    print("Loading embedding files..")
    with open(dump_dir + "/word2id.pkl", 'rb+') as f:
        word2id = pickle.load(f)
    with open(dump_dir + "/id2word.pkl", 'rb+') as f:
        id2word = pickle.load(f)
    with open(dump_dir + "/embeddings.pkl", 'rb+') as f:
        embeddings = pickle.load(f)
    print("done!")
    return word2id, id2word, embeddings

def generate_input_files(bbox_file, caption_file, out_file, word2id, N_SAMPLES=1):
    print("Generating input files using N_SAMPLES = ", N_SAMPLES)
    out = []
    with open(bbox_file, 'r') as bfile, open(caption_file, 'r') as cfile, open(out_file, 'w+') as f:
        bjs = json.loads(bfile.read())
        #cjs = json.loads(cfile.read())
        image_ids = [image['id'] for image in bjs['images']]
        categories = {cat['id']: cat['name'] for cat in bjs['categories']}
        cat2word = dict()
        for ci in categories:
            if not (categories[ci] in word2id):
                if ''.join(categories[ci].split()) in word2id:
                    cat2word[categories[ci]] = ''.join(categories[ci].split())
                elif categories[ci].split()[1] in word2id:
                    cat2word[categories[ci]] = categories[ci].split()[1]
            else:
                cat2word[categories[ci]] = categories[ci]
        if path.exists('../glove/dump/tag2ids.pkl'):
            print("Loading tag2ids..")
            with open('../glove/dump/tag2ids.pkl', 'rb+') as fdump:
                tag2ids = pickle.load(fdump)
            print("done!")
        else:
            tag2ids = dict()
            for in_img in tqdm(image_ids):
                in_annotations = [a for a in bjs['annotations'] if a['image_id'] == in_img]
                for a in in_annotations:
                    wi_id = a['category_id']
                    if wi_id in tag2ids:
                        tag2ids[wi_id] += [in_img]
                    else:
                        tag2ids[wi_id] = [in_img]
            # with open('../glove/dump/tag2ids.pkl', 'wb+') as fdump:
            #     pickle.dump(tag2ids, fdump)
        idx = 0
        for in_img in tqdm(image_ids):
            if idx > 0 and idx % 1000 == 0:
                shuffle(out)
                outline = "\n".join(out)
                f.write(outline)
                out = []
            idx += 1
            in_annotations = [a for a in bjs['annotations'] if a['image_id'] == in_img]
            for a in in_annotations:
                wi_id, wi = a['category_id'], categories[a['category_id']]
                bbox_in = a['bbox']
                if N_SAMPLES > 1:
                    for out_img in sample(tag2ids[wi_id], N_SAMPLES):
                        out_annotations = [a for a in bjs['annotations'] if a['image_id'] == out_img]
                        #out_words = [categories[a['category_id']] for a in out_annotations]
                        for aa in out_annotations:
                            wo_id, wo = aa['category_id'], categories[aa['category_id']]
                            if wi_id != wo_id:
                                bbox_out = aa['bbox']
                                # if " " in wi or " " in wo:
                                # print([image['file_name'] for image in bjs['images'] if image['id'] == in_img][0])
                                # print([image['file_name'] for image in bjs['images'] if image['id'] == out_img][0])
                                # print(wi, wo)
                                # print(bbox_in)
                                # print(bbox_out)
                                # input()
                                out_list = [in_img, out_img, word2id[cat2word[wi]], word2id[cat2word[wo]], bbox_in, bbox_out]
                                #print(out_list)
                                out += ["\t".join(map(str, out_list))]
                                #print(out)
                                #input()
                else:
                    for aa in in_annotations:
                        wo_id, wo = aa['category_id'], categories[aa['category_id']]
                        if wi_id != wo_id:
                            bbox_out = aa['bbox']
                            out_list = [in_img, in_img, word2id[cat2word[wi]], word2id[cat2word[wo]], bbox_in, bbox_out]
                            out += ["\t".join(map(str, out_list))]
        shuffle(out)
        outline = "\n".join(out)
        f.write(outline)


def generate_input_files_with_verbs(bbox_file, caption_file, out_file, word2id):
    out = generate_input_files(bbox_file, caption_file, out_file, word2id)
    # TODO: Generate verbs and add them in out
    outline = "\n".join(out)
    with open(out_file, 'w+') as f:
        f.write(outline)


if __name__ == '__main__':
    bbox_file = '../../DATA/annotations/instances_train2014.json'
    caption_file = '../../DATA/annotations/annotations_trainval2014/captions_train2014.json'
    out_file = '../DATA/train2014.tsv'
    glove_file = '../glove/glove.840B.300d.txt'
    dump_dir = '../glove/dump'
    #compute_embeddings_and_dictionaries(glove_file, dump_dir)
    word2id, id2word, embeddings = load_embeddings_and_dictionaries(dump_dir)
    #generate_input_files(bbox_file, caption_file, out_file, word2id)
