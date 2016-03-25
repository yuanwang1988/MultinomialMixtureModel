function triplet = mat2triplets(matrix)
%MAT2TRIPLETS Summary of this function goes here
%   Purpose: convert a NxM matrix into a Tx3 matrix of triplets where T is
%   the number of non-zero elements in the matrix
%
%   Input:
%       - matrix - NxM matrix
%   Ouput:
%       - triplet - Tx3 matrix where T is the number of non-zero elements
%       of the matrix. The first column is row index, the second column is
%       column index and the third column is the value.
        

[N, M] = size(matrix);
nonZeroCount = nnz(matrix);

triplet = zeros(nonZeroCount, 3);


triplet_idx = 1
for i = 1:N
    for j = 1:M
        if matrix(i,j) ~= 0
            triplet(triplet_idx, 1) = i;
            triplet(triplet_idx, 2) = j;
            triplet(triplet_idx, 3) = matrix(i,j);
            triplet_idx = triplet_idx+1;
            
        end
    end
end


end

