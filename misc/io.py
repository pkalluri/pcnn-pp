import pickle
from inspect import currentframe, getframeinfo


def save_pickle(obj, path):
    pickle.dump( obj, open( path, 'wb' ) )

def load_pickle(path):
    return pickle.load( open( path, 'rb'))

def log(s):
    frame = currentframe()
    filename = getframeinfo(frame).filename
    line_number = frame.f_back.f_lineno
    print('{fn} {no}: {s}'.format(fn=filename, no=line_number, s=s))