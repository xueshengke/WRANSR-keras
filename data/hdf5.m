clear; clc;

filepath = './train/train_wavelet_x4.h5';

h5disp(filepath)
data = h5read(filepath, '/data');
label = h5read(filepath, '/label');
size(data)
size(label)
data = permute(data, [4, 3, 2, 1]);
label = permute(label, [4, 3, 2, 1]);
chunksz = 32;
data_dim = size(data)
label_dim = size(label)

new_h5_file = 'train_wavelet_x4.h5';
h5create(new_h5_file, '/data', data_dim, ...
         'Datatype', 'single', 'ChunkSize', [data_dim(1:end-1) chunksz]);
h5create(new_h5_file, '/label', label_dim, ...
         'Datatype', 'single', 'ChunkSize', [data_dim(1:end-1) chunksz]);
h5write(new_h5_file, '/data', data);
h5write(new_h5_file, '/label', label);

h5disp(new_h5_file)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filepath = './test/test_wavelet_x4.h5';

h5disp(filepath)
data = h5read(filepath, '/data');
label = h5read(filepath, '/label');
size(data)
size(label)
data = permute(data, [4, 3, 2, 1]);
label = permute(label, [4, 3, 2, 1]);
chunksz = 2;
data_dim = size(data)
label_dim = size(label)

new_h5_file = 'test_wavelet_x4.h5';
h5create(new_h5_file, '/data', data_dim, ...
         'Datatype', 'single', 'ChunkSize', [data_dim(1:end-1) chunksz]);
h5create(new_h5_file, '/label', label_dim, ...
         'Datatype', 'single', 'ChunkSize', [data_dim(1:end-1) chunksz]);
h5write(new_h5_file, '/data', data);
h5write(new_h5_file, '/label', label);

h5disp(new_h5_file)
