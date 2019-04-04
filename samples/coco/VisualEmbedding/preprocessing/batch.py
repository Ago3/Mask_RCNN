import ast
import numpy as np


class Batch(object):
    """Batch"""
    def __init__(self, batch_size, I_in, I_out, W_in, W_out, Bbox_in, Bbox_out):
        self.size = batch_size
        self.I_in = I_in
        self.I_out = I_out
        self.W_in = np.array(W_in)
        self.W_out = np.array(W_out)
        self.Bbox_in = np.array(Bbox_in)
        self.Bbox_out = np.array(Bbox_out)


#bjs = json.loads(bfile.read())
#object_categories should be a list [idx in dictionary of 1st category, idx in dictionary of 2nd category..]
def generate_batch_coco(input_file, object_categories, batch_size, img2vec=None):
    with open(input_file, 'r') as f:
        last_batch = False
        end_epoch = False
        while not last_batch:
            if end_epoch:
                batch = None
            else:
                I_in_ids, I_out_ids, W_in, W_out, Bbox_in, Bbox_out = [], [], [], [], [], []
                while len(I_in_ids) < batch_size:
                    line = f.readline()
                    if not line:
                        f.seek(0)
                        last_batch = True
                    else:
                        #line format: s = "1 2 3 4 [1,2,3,4] [2,3,4,5]"
                        img_in, img_out, w_in, w_out, bbox_in, bbox_out = [ast.literal_eval(e) for e in line.split()]
                        I_in_ids += [img_in]
                        I_out_ids += [img_out]
                        W_in += [w_in]
                        W_out += [w_out]
                        Bbox_out += [bbox_out]
                I_in = np.array([img2vec.get_features(i[0]) for i in I_in_ids])
                I_out = np.array([img2vec.get_features(i[0]) for i in I_out_ids])
                batch = Batch(batch_size, I_in, I_out, W_in, W_out, Bbox_in, Bbox_out)
            yield batch
            end_epoch = last_batch
            last_batch = False
