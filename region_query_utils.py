import numpy as np
import nengo_spa as spa

from utils import power, encode_point, spatial_dot

from scipy.signal import correlate


#returns whether x,y is positive, use directly indexing 2x2 quadrant region selector
def direction_quad(x,y):
    x_ = np.array(x) >= 0
    y_ = np.array(y) >= 0
    return tuple(np.stack([x_,y_],1).astype(int).T)

#closed form function to generate rectangular region SSP
#ranges are 2-tuples, X & Y are axis ssps
def generate_rectangle_region(x_range, y_range, X, Y):
    #integrating eulor's formula e^ikx from a to b is i(e^ika - e^ikb)/k where e^ik is the axis vector and k is the angle of the vector
    fft_X = np.fft.fft(X.v)
    fft_Y = np.fft.fft(Y.v)

    phi = np.angle(fft_X)
    gamma = np.angle(fft_Y)
    assert np.allclose(np.abs(fft_X), 1)
    assert np.allclose(np.abs(fft_Y), 1)

    #masks to seperate when angle is 0, then you're integrating a constant so it's simply b-a
    phi_mask = phi != 0
    gamma_mask = gamma != 0

    x_integral = np.zeros_like(phi)
    y_integral = np.zeros_like(phi)

    x_integral[phi_mask] = np.fft.fft((power(X, x_range[1]) - power(X, x_range[0])).v)[phi_mask] * 1j / phi[phi_mask]
    x_integral[np.logical_not(phi_mask)] = x_range[1]-x_range[0]

    y_integral[gamma_mask] = np.fft.fft((power(Y, y_range[1]) - power(Y, y_range[0])).v)[gamma_mask] * 1j / gamma[gamma_mask]
    y_integral[np.logical_not(gamma_mask)] = y_range[1]-y_range[0]

    rectangle = spa.SemanticPointer(np.fft.ifft(x_integral * y_integral))
    return rectangle

#generate space lookup table
def generate_space_table(xs,ys,dim,X,Y):
    vs = np.zeros((len(xs),len(ys),dim))
    for i,x in enumerate(xs):
        for j, y in enumerate(ys):
            vs[i,j,:] = (power(X, x)*power(Y,y)).v
    return vs

# find vector coordinates from lookup table
def lookup_space_table(loc, table):
    if isinstance(loc, spa.SemanticPointer):
        loc = loc.v
    dots = np.sum(table * loc[None, None, :], axis = 2)
    ind = np.unravel_index(np.argmax(dots, axis=None), dots.shape)
    loc = table[ind]
    return loc #, ind

#limit gets doubled for shifting
def get_quads(X,Y, limit):
    UP_RIGHT = generate_rectangle_region([0,limit*2], [0, limit*2], X, Y)
    DOWN_RIGHT = generate_rectangle_region([0,limit*2], [-limit*2, 0], X, Y)
    UP_LEFT = generate_rectangle_region([-limit*2, 0], [0, limit*2], X, Y)
    DOWN_LEFT = generate_rectangle_region([-limit*2, 0], [-limit*2, 0], X, Y)
    return UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT

#blur the image
def squint(A,k):
    return correlate(A,k, mode='same')

def get_max_coords(A):
    ind = np.unravel_index(np.argmax(A, axis=None), A.shape)
    return ind

def ignore(A, mask):
    return A*mask

def update_mask(mask, ind, radius, eps = 0):
    mask = np.ones(mask.shape)*eps+mask*(1-eps)
    mask[ind[0]-radius:ind[0]+radius,ind[1]-radius:ind[1]+radius] = 0
    return mask

def saccades(images, filter = None, eps = 0):
    if filter is None:
        k = np.ones((14,14))
        filter = correlate(k,k)
    xs = []
    ys = []
    for i in range(len(images)):
        mask = np.ones((120,120))
        img = images[i]
        x = []
        y = []
        for i in range(4):
            blurred = squint(img*mask, filter)
            x_,y_ = get_max_coords(blurred)
            mask = update_mask(mask, [x_,y_], 14, eps)
            x.append(x_)
            y.append(y_)
        xs.append(x)
        ys.append(y)
    return xs, ys

#predictions for queries with single direction from single object
def predict_single_query(obj_loc_memory, obj_memory, query_obj, dirs, obj_dic, region_selector, loc_table):
    try: #in case of single query
        n = len(query_obj)
    except Exception:
        n = 1
        query_obj = [query_obj]
        obj_loc_memory = [obj_loc_memory]
        obj_memory = [obj_memory]
        dirs = [dirs] #might not need this one

    extracted_query_loc = [obj_loc_memory[i] * ~query_obj[i] for i in range(n)]
    extracted_query_loc = [spa.SemanticPointer(lookup_space_table(V, loc_table)) for V in extracted_query_loc]

    dir_regions = region_selector[dirs]
    query_region = [extracted_query_loc[i] * dir_regions[i] for i in range(n)]
    extract = np.array([obj_loc_memory[i] * ~ query_region[i] for i in range(n)])

    dots = np.array([obj_dic.dot(_) for _ in extract])

    # object_memory_raw = np.sum(obj_memory, axis=1)
    object_memory_raw = obj_memory
    object_memory = object_memory_raw - query_obj #eliminate one instance of queried object
    extract_objs = np.array([obj_dic.dot(_) > 0.8 for _ in object_memory])

    preds = np.where(extract_objs, dots, -1) #set non-present objects to -1
    obj_preds = np.argmax(preds, axis = 1)
    return obj_preds #, preds #return similarities too 



#predictions for queries with single direction from single object
def predict_single_query_threshold(obj_loc_memory, obj_memory, query_obj, dirs, obj_dic, region_selector, loc_table):
    try: #in case of single query
        n = len(query_obj)
    except Exception:
        n = 1
        query_obj = [query_obj]
        obj_loc_memory = [obj_loc_memory]
        obj_memory = [obj_memory]
        dirs = [dirs] #might not need this one

    extracted_query_loc = [obj_loc_memory[i] * ~query_obj[i] for i in range(n)]
    extracted_query_loc = [spa.SemanticPointer(lookup_space_table(V, loc_table)) for V in extracted_query_loc]

    dir_regions = region_selector[dirs]
    query_region = [extracted_query_loc[i] * dir_regions[i] for i in range(n)]
    extract = np.array([obj_loc_memory[i] * ~ query_region[i] for i in range(n)])

    dots = np.array([obj_dic.dot(_) for _ in extract])

    # object_memory_raw = np.sum(obj_memory, axis=1)
    object_memory_raw = obj_memory
    object_memory = object_memory_raw - query_obj #eliminate one instance of queried object
    extract_objs = np.array([obj_dic.dot(_) > 0.8 for _ in object_memory])

    preds = dots * extract_objs
    obj_preds = np.argmax(preds, axis = 1)
    pred_value = np.max(preds, axis = 1)
    return obj_preds, pred_value

#predicting single object from query using 2 objects 
def predict_double_query(obj_loc_memory, obj_memory, query_obj, dirs, query_obj2, dirs2, obj_dic, region_selector, loc_table):
    try: #in case of single query
        n = len(query_obj)
    except Exception:
        n = 1
        query_obj = [query_obj]
        query_obj2 = [query_obj2]
        obj_loc_memory = [obj_loc_memory]
        obj_memory = [obj_memory]
        dirs = [dirs] #might not need this one
        dirs2 = [dirs2] #might not need this one

    extracted_query_loc = [obj_loc_memory[i] * ~query_obj[i] for i in range(n)]
    extracted_query_loc = [spa.SemanticPointer(lookup_space_table(V, loc_table)) for V in extracted_query_loc]

    extracted_query_loc2 = [obj_loc_memory[i] * ~query_obj2[i] for i in range(n)]
    extracted_query_loc2 = [spa.SemanticPointer(lookup_space_table(V, loc_table)) for V in extracted_query_loc2]

    dir_regions = region_selector[dirs]
    dir_regions2 = region_selector[dirs2]
    query_region = [extracted_query_loc[i] * dir_regions[i] + extracted_query_loc2[i] * dir_regions2[i]  for i in range(n)]
    extract = np.array([obj_loc_memory[i] * ~ query_region[i] for i in range(n)])

    dots = np.array([obj_dic.dot(_) for _ in extract])

    # object_memory_raw = np.sum(obj_memory, axis=1)
    object_memory_raw = obj_memory
    object_memory = object_memory_raw - query_obj #eliminate one instance of queried object
    extract_objs = np.array([obj_dic.dot(_) > 0.8 for _ in object_memory])

    preds = dots * extract_objs
    obj_preds = np.argmax(preds, axis = 1)
    return obj_preds

#old generate function
def generate_rectangle_region_old(x_range, y_range, X, Y, resolution = 100):
    fft_X = np.fft.fft(X.v)
    fft_Y = np.fft.fft(Y.v)

    phi = np.angle(fft_X)
    gamma = np.angle(fft_Y)
    assert np.allclose(np.abs(fft_X), 1)
    assert np.allclose(np.abs(fft_Y), 1)
    if any(phi == 0):
        # can't divide, just use summation
        region_analytic = np.zeros_like(X.v)
        for x in np.linspace(*x_range, resolution):
            for y in np.linspace(*y_range, resolution):
                region_analytic += encode_point(x, y, X, Y).v
        return spa.SemanticPointer(region_analytic/np.max(spatial_dot(region_analytic, np.linspace(*x_range,resolution/5), np.linspace(*y_range,resolution/5),X, Y)))
    else:
        # (FYI this is Euler's formula as we are applying it implicitly)
        # pi = phi * x1
        # assert np.allclose(fft_X ** x1, np.cos(pi) + 1j * np.sin(pi))
        INVPHI = spa.SemanticPointer(np.fft.ifft(1j / phi))
        INVGAMMA = spa.SemanticPointer(np.fft.ifft(1j / gamma))

        region_algebraic = (((power(X, x_range[1]) - power(X, x_range[0])) * INVPHI) *
                            (((power(Y, y_range[1]) - power(Y, y_range[0])) * INVGAMMA)))
        return region_algebraic
