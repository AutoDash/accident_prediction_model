import os
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence

class ZipGenerator(Sequence):
    def __self__(self, zfile, path='/'):
        if path[0] == '/':
            path = path[1:]
        self.zfile = zfile
        zfs = ZipFile(self.zfile)
        self.elements = [
            fs for fs in zfs.infolist()
            if not fs.is_dir() and fs.filename.startswith(path)
        ]

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        batch_file = self.elements[idx]
        zfs = ZipFile(self.zfile)
        zfs.extract(self.elements[idx], path='data')
        batch_data = np.load(f'data/{batch_file.filename}')
        X = batch_data['data']
        Y = batch_data['labels']
        os.remove(f'data/{batch_file.filename}')
        return (X, Y)
