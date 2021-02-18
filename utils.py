from os import path
import pickle

COLOR_GREEN = (0, 250, 0)

def load_pickle(filepath):
    with open(filepath,'rb') as f:
        return pickle.load(f)

def dump_file_as_pickle(obj, filepath):
    with open(filepath,'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath

if __name__ == '__main__':
    
    a = load_pickle('pose_data.pickle')
    print('here')