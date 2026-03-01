rng(1)

n = 100;

c = randn(1,n);
r = [c(1) randn(1,n-1)];
A = toeplitz(c,r);

% Construct the basis
for i = -n+1:n-1
    P(:,:,n+i) = diag(ones(n-abs(i),1), i);
end


for k = 1:size(P,3)
    P(:,:,k) = P(:,:,k) / norm(P(:,:,k), 'fro');
end

[Delta,iteration_count] = structured_distance_to_singularity(A,P);

% Construct the singular matrix
AplusD = A+Delta;

% Check that the output is indeed singular
[U,S,V] = svd(AplusD);

disp(['Norm of the input matrix is ' num2str(norm(A,'f'))])
disp(['The computed distance is ' num2str(norm(Delta,'f'))])
disp(['The smallest singular value is ' num2str(S(end,end))])

