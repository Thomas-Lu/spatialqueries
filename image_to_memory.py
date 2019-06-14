import nengo
import numpy as np
import pickle
import keras
from keras import backend as K
import nengo
from utils import encode_point
import argparse

def decode_image_uncentered(images, xs, ys, im_dim, model):
    pred_obj_list = []

    for i, image in enumerate(images):
        img = np.array([np.expand_dims(np.array(image[x_:x_+im_dim, y_:y_+im_dim]), axis=2) for x_,y_ in zip(xs[i], ys[i])])
        # print(img.shape)
        pred = np.argmax(model.predict(img), axis=1)
        # print(pred)
        # print(obj_list[i])
        pred_obj_list.append(pred)

    pred_obj_list = np.array(pred_obj_list)
    return pred_obj_list

def decode_image(images, xs, ys, rad, model):
    pred_obj_list = []

    for i, image in enumerate(images):
        img = np.array([np.expand_dims(np.array(image[x_-rad:x_+rad, y_-rad:y_+rad]), axis=2) for x_,y_ in zip(xs[i], ys[i])])
        # print(img.shape)
        pred = np.argmax(model.predict(img), axis=1)
        # print(pred)
        # print(obj_list[i])
        pred_obj_list.append(pred)

    pred_obj_list = np.array(pred_obj_list)
    return pred_obj_list

def encode_memory(pred_obj_list, xs, ys, obj_vectors, axis_vec, n,m,size = 120, lim = 5):
    individual_obj_vectors = obj_vectors[pred_obj_list]
    scale = 120/(lim*2)

    loc_vectors = np.array([encode_point(x/scale -lim,y/scale -lim, axis_vec[0], axis_vec[1])
        for x,y in zip(np.array(xs).ravel(), np.array(ys).ravel())]).reshape(n,m)

    encoded_objs = individual_obj_vectors*loc_vectors

    obj_loc_memory = np.sum(encoded_objs, axis=1)
    obj_memory = np.sum(individual_obj_vectors, axis=1)

    memory_data = {}
    memory_data['obj_loc_memory'] = obj_loc_memory
    memory_data['obj_memory'] = obj_memory
    memory_data['individual_obj_vectors'] = individual_obj_vectors
    return memory_data

def encode_memory_shape(pred_obj_list, xs, ys, obj_vectors, axis_vec,shape,n,m, size = 120, lim = 5 ):
    individual_obj_vectors = obj_vectors[pred_obj_list]
    scale = 120/(lim*2)

    loc_vectors = np.array([encode_point(x/scale -lim,y/scale -lim, axis_vec[0], axis_vec[1])
        for x,y in zip(np.array(xs).ravel(), np.array(ys).ravel())]).reshape(n,m)

    encoded_objs = individual_obj_vectors*loc_vectors*shape

    obj_loc_memory = np.sum(encoded_objs, axis=1)
    obj_memory = np.sum(individual_obj_vectors, axis=1)

    memory_data = {}
    memory_data['obj_loc_memory'] = obj_loc_memory
    memory_data['obj_memory'] = obj_memory
    memory_data['individual_obj_vectors'] = individual_obj_vectors
    return memory_data
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--imdim', type=int, default=28)
    parser.add_argument('--savefile', type=str, default='data512')
    parser.add_argument('--imagefile', type=str, default='generated_images')
    parser.add_argument('--vectorfile', type=str, default='image_and_memory')
    parser.add_argument('--modelfile', type=str, default='mnist_net')
    args = parser.parse_args()

    n = args.n
    m = args.m
    im_dim = args.imdim
    savefile = args.savefile
    imagefile = args.imagefile
    vectorfile = args.vectorfile
    modelfile = args.modelfile

    # Get a batch of n random images with m digits
    model = keras.models.load_model(modelfile+'.h5')


    objs = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]

    mnist_datafile = imagefile+'.p'
    img_data = pickle.load(open(mnist_datafile,'rb'))
    images = img_data['images']
    xs = img_data['x']
    ys = img_data['y']
    obj_list = img_data['obj_list']

    pred_obj_list = decode_image(images, xs, ys, im_dim, model)

    spa_datafile = vectorfile+'.p'
    spa_data = pickle.load(open(spa_datafile,'rb'))
    axis_vec = spa_data['axis_vec']
    obj_dict = spa_data['obj_dict']

    objs = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    obj_vectors = np.stack([obj_dict[_] for _ in objs])

    size = 120
    lim = 5

    memory_data = encode_memory(pred_obj_list,xs,ys,obj_vectors,axis_vec,n,m)
    memory_data['obj_vectors'] = obj_vectors
    memory_data['objs'] = objs
    memory_data['axis_vec'] = axis_vec
    memory_data['obj_dict'] = obj_dict
    memory_data['pred_obj_list'] = pred_obj_list

    pickle.dump(memory_data, open(savefile+".p", "wb"))


if __name__ == '__main__':
    main()