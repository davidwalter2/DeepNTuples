

import h5py
h5f = h5py.File(fileprefix, 'w')

#Shuffle two files
def shuffleFiles(file1, file2):






def writeOut(self, file):
    import h5py
    import numpy


    h5f1 = h5py.File(file, 'w')

    # try "lzf", too, faster, but less compression
    def _writeoutListinfo(arrlist, fidstr, h5F):
        arr = numpy.array([len(arrlist)])
        h5F.create_dataset(fidstr + '_listlength', data=arr)
        for i in range(len(arrlist)):
            idstr = fidstr + str(i)
            h5F.create_dataset(idstr + '_shape', data=arrlist[i].shape)

    def _writeoutArrays(arrlist, fidstr, h5F):
        for i in range(len(arrlist)):
            idstr = fidstr + str(i)
            arr = arrlist[i]
            h5F.create_dataset(idstr, data=arr, compression="lzf")

    arr = numpy.array([self.nsamples], dtype='int')
    h5f.create_dataset('n', data=arr)

    _writeoutListinfo(self.w, 'w', h5f)
    _writeoutListinfo(self.x, 'x', h5f)
    _writeoutListinfo(self.y, 'y', h5f)

    _writeoutArrays(self.w, 'w', h5f)
    _writeoutArrays(self.x, 'x', h5f)
    _writeoutArrays(self.y, 'y', h5f)

    h5f.close()

