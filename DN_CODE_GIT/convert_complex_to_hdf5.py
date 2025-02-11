import os
from tkinter import filedialog
from tkinter import *
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
root = Tk()
root.withdraw()

folder_of_data = filedialog.askdirectory()

out_file_name = os.path.join(folder_of_data,'C.h5')

comp_type = np.dtype([('Orbit', 'i'), ('Location', np.str_, 6), ('Temperature (F)', 'f8'), ('Pressure (inHg)', 'f8')])

with h5.File(out_file_name, 'w') as hf:
    grp = hf.create_group("Images")
    for ch in range(16):
        name = os.path.join( folder_of_data, f'X_000_{ch:03}.dat.complex')
        print(f'Read {name}')
        raw = np.fromfile(name,dtype=np.complex64)
        raw = np.reshape(raw,(256,192,256))

        plt.figure()
        plt.imshow(np.angle(raw[128,]))
        plt.show()

        case_name = f'C_000_{ch:03}'
        grp.create_dataset(case_name, data=raw)

    grpinfo = hf.create_group("Info")
    grpinfo.attrs.create("rcencodes", int(999))





