import numpy as np
import nengo
from sklearn.metrics import mean_squared_error
import nengo_spa as spa
import warnings

def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)

	
#rewrite for n-dim points (e.g. 5 dim for x,y, r,g,b)
def encode_point(x, y, x_axis, y_axis):
    return power(x_axis, x) * power(y_axis, y)


def spatial_dot(vec, xs, ys, x_axis, y_axis, swap = False):
    if isinstance(vec, spa.SemanticPointer):
        vec = vec.v
    vs = np.zeros((len(ys), len(xs)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis=x_axis, y_axis=y_axis,
            )
            vs[j, i] = np.dot(vec, p.v)
    if swap:
        vs = np.transpose(vs)
    return vs

# make good unitary that behaves better with exponents but doesn't behave well with closed for rectangle function
def make_good_unitary(dim, eps=1e-3, rng=np.random):
    # created by arvoelke
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v).unitary()

#make good unitary, compatible with closed form rectangle
def make_good_unitary_old(D, eps=np.pi*1e-3, n_trials=10000):
    for _ in range(n_trials):
        d = spa.Vocabulary(D)
        sp = d.create_pointer().unitary()
        a = np.angle(np.fft.fft(sp.v))
        if np.all(np.abs(a) > eps):
            return spa.SemanticPointer(sp.v)
    raise RuntimeError("bleh")

#create object and axis vectors
# def create_vectors(objs, D, axis = {'X', 'Y'}):
#     init_dic = spa.Vocabulary(dimensions = D, max_similarity = 0.01)
#     vec_dic = {}

#     for a in axis:
#         ax = make_good_unitary(D)

#         init_dic.add(a, ax)
#         vec_dic[a] = ax

#     # for item in vecs:
#     #     init_dic.add(item, init_dic.create_pointer(unitary =True, attempts=5000))

#     #using one dictionary for both to reduce similarity
#     for item in objs:
#         init_dic.add(item, init_dic.create_pointer(attempts=5000))

#     obj_dic = spa.Vocabulary(dimensions = D, max_similarity = 0.01)
#     for item in objs:
#         obj_dic.add(item, init_dic[item].v)

#     return obj_dic, vec_dic

def create_vectors(objs, D, axis = {'X', 'Y'}, max_similarity=0.01, attempts = 1000):
    init_dic = spa.Vocabulary(dimensions = D, max_similarity = max_similarity)
    for a in axis:
        if len(init_dic) > 0:
            for i in range(attempts):
                ax = make_good_unitary(D)
                if np.max(np.abs(init_dic.dot(ax))) < 0.01:
                    init_dic.add(a, ax.v)
                    break
            else:
                warnings.warn(
                        'Could not create a semantic pointer with '
                        'max_similarity=%1.2f'
                        % (max_similarity))
        else:
            init_dic.add(a, make_good_unitary(D))

    for item in objs:
        for i in range(attempts):
            v = np.random.randn(D)
            v /= np.linalg.norm(v)
            if np.max(np.abs(init_dic.dot(v))) < max_similarity:
                init_dic.add(item, v)
                break
        else:
            warnings.warn(
                'Could not create a semantic pointer with '
                'max_similarity=%1.2f'
                % (max_similarity))

    vec_dic =  spa.Vocabulary(dimensions = D, max_similarity = max_similarity)
    obj_dic =  spa.Vocabulary(dimensions = D, max_similarity = max_similarity)

    for a in axis:
        vec_dic.add(a, init_dic[a].v)
    
    for item in objs:
        obj_dic.add(item, init_dic[item].v)

    return obj_dic, vec_dic

def generate_item_memory(dim, n_items, limits, x_axis_vec, y_axis_vec, normalize_memory=True):
    """
    Create a semantic pointer that contains a number of items bound with respective coordinates
    Returns the memory, along with a list of the items and coordinates used
    """

    # Start with an empty memory
    memory_sp = spa.SemanticPointer(data=np.zeros((dim)))
    coord_list = []
    item_list = []

    for n in range(n_items):
        # Generate random point
        x = np.random.uniform(low=limits[0], high=limits[1])
        y = np.random.uniform(low=limits[2], high=limits[3])
        pos = encode_point(x, y, x_axis=x_axis_vec, y_axis=y_axis_vec)

        # Generate random item
        item = spa.SemanticPointer(dim)

        # Add the item at the point to memory
        memory_sp += (pos * item)

        coord_list.append((x, y))
        item_list.append(item)

    if normalize_memory:
        memory_sp.normalize()

    return memory_sp, coord_list, item_list


def spatial_plot(vs, colorbar=True, vmin=-1, vmax=1, cmap='plasma'):
    vs = vs[::-1, :]
    plt.imshow(vs, interpolation='none', extent=(xs[0],xs[-1],ys[0],ys[-1]), vmax=vmax, vmin=vmin, cmap=cmap)
    if colorbar:
        plt.colorbar()