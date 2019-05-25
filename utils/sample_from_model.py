import numpy as np
import sys


def sample_from_model(sess, obs_shape, new_x_gen, xs, batch_size_generator, nr_gpu):
    x_gen = [np.zeros((batch_size_generator,) + obs_shape, dtype=np.float32) for i in range(nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(nr_gpu)})
            sys.stdout.write(".")
            sys.stdout.flush()
            for i in range(nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    print()
    return np.concatenate(x_gen, axis=0)