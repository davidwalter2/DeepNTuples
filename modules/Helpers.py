import numpy as np

def shuffle_in_unison(listofArrays):
    '''
    :param list of arrays: a list of numpy arrays with the same length in the first dimension
    :return: list of arrays shuffled only in the first dimension
    '''
    shuffled_list = []
    for arr in listofArrays:
        shuffled_list.append(np.empty(arr.shape, dtype=arr.dtype))
    permutation = np.random.permutation(len(listofArrays[0]))
    for old_index, new_index in enumerate(permutation):
        for i, arr in enumerate(listofArrays):
            shuffled_list[i][new_index] = arr[old_index]
    return shuffled_list