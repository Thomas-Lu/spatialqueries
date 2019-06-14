# Tested with Python 3.5.2 with tensorflow and matplotlib installed.
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


def display_image(arr):
    # dim = np.int(np.sqrt(arr.shape[0]))
    # arr = np.reshape(arr, (dim, dim)) * 255).astype(np.uint8)
    plt.imshow(arr, interpolation='nearest')
    return plt

def placement(nums, vals, m=3, size = 120, spread = 4, dim = 28, colour = False):
    if colour:
        placed = np.zeros((size, size, 3))
    else:
        placed = np.zeros((size, size))
    
        scale = (size-dim)//spread
        xs = scale//4 + np.random.choice(spread,m, replace=False)*scale + np.random.choice(scale//2, m)
        ys = scale//4 + np.random.choice(spread,m, replace=False)*scale + np.random.choice(scale//2, m)

    # place_cats = np.zeros((size,size,10))
    for i, img in enumerate(nums):
        img = np.reshape(img, (dim, dim))
        x_ = xs[i]
        y_ = ys[i]
        tmp = np.zeros((size, size))
        tmp[x_:x_+dim, y_:y_+dim] = img
        # place_cats[x_:x_+dim, y_:y_+dim, vals[i]] = img > 0.1
        if colour:
            col = np.random.random_sample(3)
            tmp = col*np.stack([tmp, tmp, tmp], axis =2)
        placed = np.maximum(placed, tmp)
    return placed, xs, ys #, place_cats #place cats for placing images in seperate category channels


def gen_images(n,m, random_m = False, display = False):
    #mnist dataset stuff

    mnist = tf.keras.datasets.mnist
    train, test = mnist.load_data()
    trainx, trainy = train
    trainx = trainx.astype('float32')/255
    dataset_length = len(train[0])

    images = []
    x_ = []
    y_ = []
    obj_list = []
    m_list = []
    image_categories = []

    for i in range(n):
        _m = 0
        xs_list = []
        objs = []
        if random_m:
            m1 = np.random.random_integers(*m)
        else:
            m1 = m
        while _m < m1:
            # xs, ys = mnist.test.next_batch(1)
            # y = np.argmax(ys)
            sample = np.random.choice(dataset_length)
            xs, y = trainx[sample], trainy[sample]
            while(y in objs):
                sample = np.random.choice(dataset_length)
                xs, y = trainx[sample], trainy[sample]
                # xs, ys = mnist.test.next_batch(1)
                # y = np.argmax(ys)
            
            xs_list.append(xs)
            # xs_list.append(np.array(xs))
            objs.append(y)
            _m += 1
                
        # batch_xs = np.vstack(xs_list)
        batch_xs = np.array(xs_list)
        image, xs, ys = placement(batch_xs, objs, m=m1)
        if display:
            display_image(image).show()
            print(objs)
            print(xs, ys)
        images.append(image)
        x_.append(xs)
        y_.append(ys)
        obj_list.append(objs)
        m_list.append(m1)
        # image_categories.append(cat_image)

    # data = {'images':images, 'x':x_, 'y':y_, 'obj_list':obj_list, 'm_list':m_list, 'image_categories' : image_categories}
    data = {'images':images, 'x':x_, 'y':y_, 'obj_list':obj_list}

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--file', type=str, default='generated_images_testing')
    args = parser.parse_args()

    n = args.n
    m = args.m
    savefile = args.file

    # Get a batch of n random images with m digits
    # data = gen_images(n,m = (2,4), random_m = True) #range of m
    data = gen_images(n ,m = 4, random_m = False)
    pickle.dump(data, open(savefile+".p", "wb"))


if __name__ == '__main__':
    main()