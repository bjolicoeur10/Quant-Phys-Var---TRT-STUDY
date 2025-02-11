import numpy as np
import random
import h5py

file_path = r'Z:\01_AUTORECON_INBOX\trtvol9_00560_2023-06-21\00560_00003_pcvpir_flow_repeat\raw_data'
out_path = r'E:\Data\dlv\INTER_LEAF\trt_intleaf\vol9\2'
main_file = 'Images.h5'
encode_split_type = 'normal'  # current inputs 'random' , 'mixed',  or 'normal


def is_dataset_in_h5(filename, dataset_name):
    with h5py.File(filename, 'r') as f:
        if dataset_name in f:
            return True
        else:
            return False


filename = f'{file_path}\{main_file}'
dataset_name = '/Images/Encode_000_Frame_000'
encode_index = 0
while is_dataset_in_h5(filename, dataset_name):
    dataset_name = f'/Images/Encode_{encode_index:03}_Frame_000'
    encode_index += 1
num_encodes = encode_index - 1
print(f'number of encodes is {num_encodes}')
dataset_name = '/Images/Encode_000_Frame_000'
frame_index = 0
while is_dataset_in_h5(filename, dataset_name):
    dataset_name = f'/Images/Encode_000_Frame_{frame_index:03}'
    frame_index += 1
num_frames = frame_index - 1
print(f'number of frames is {num_frames}')
if num_frames == 1:
    print('data only has one frame')
else:
    with h5py.File(f'{file_path}\{main_file}', 'r') as file:
        with h5py.File(f'{out_path}\Images_TR_TA.h5', 'w') as f:
            group = f.create_group('/Images')
            for enci in range(num_encodes):
                IMGALL_real = []
                IMGALL_imag = []
                # running_sum_real= np.zeros((3,3,3))
                # running_sum_imaginary = np.zeros((3,3,3))
                for frai in range(num_frames):
                    data_name = f'/Images/Encode_{enci:03}_Frame_{frai:03}'
                    # dataset_name replaced with dataname

                    temp = np.array(file[data_name])
                    # IMGALL_real.append(temp['real'])
                    # IMGALL_imag.append(temp['imag'])
                    real_data = np.array(temp['real'])
                    imaginary_data = np.array(temp['imag'])
                    IMGALL_real.append(real_data)
                    IMGALL_imag.append(imaginary_data)
                    # data_type = np.dtype(dataset)
                    # data_shape = dataset.shape
                    # out_name = f'/Images/Encode_{enci:03}_Frame_{frai:03}'
                    # f.create_dataset(out_name,shape=data_shape, dtype=data_type, data=dataset)

                data_name = f'/Images/Encode_{enci:03}_Frame_000'
                dataset = group.create_dataset(data_name, shape=(320, 320, 320),
                                               dtype=np.dtype([('real', np.float32), ('imag', np.float32)]))
                running_sum_real = np.mean(IMGALL_real, axis=0)
                running_sum_imaginary = np.mean(IMGALL_imag, axis=0)
                dataset['real'] = running_sum_real
                dataset['imag'] = running_sum_imaginary

# import h5py
# import numpy as np

# # Define the filename and dataset name
# filename = 'Images.h5'
# dataset_name = 'Encode_000_Frame_000'

# # Create a new HDF5 file
# with h5py.File(filename, 'w') as f:
#     # Create the group and dataset
#     group = f.create_group('Images')
#     dataset = group.create_dataset(dataset_name, shape=(320, 320, 320), dtype=np.dtype([('real', np.float32), ('imag', np.float32)]))

#     # Write the data to the dataset
#     real_data = np.random.rand(320, 320, 320).astype(np.float32)
#     imaginary_data = np.random.rand(320, 320, 320).astype(np.float32)
#     dataset['real'] = real_data
#     dataset['imag'] = imaginary_data

#     # Set the chunk size, filters, and fill value
#     dataset.chunks = None
#     dataset.fill_value = np.nan

