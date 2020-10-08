import pickle

def save_pickle(obj, path):
    pickle.dump( obj, open( path, 'wb' ) )

def load_pickle(path):
    return pickle.load( open( path, 'rb'))