load a3dataFinal.mat

train_vec = mat2triplets(train_data);
test_vec = mat2triplets(test_data);

data_vec = [train_vec; test_vec];

N = size(data_vec, 1)

probe_idx = randperm(N, floor(N/10));

probe_vec = data_vec(probe_idx, :);
train_vec = data_vec(~ismember(1:N, probe_idx),:);

save('moviedata.mat', 'train_vec', 'probe_vec')