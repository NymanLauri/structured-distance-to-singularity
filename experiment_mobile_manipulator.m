%EXPERIMENT_MOBILE_MANIPULATOR
%
% This script compares the SVS and RO methods for the 
% 'mobile_manipulator' quadratic eigenvalue problem (QEP) from the NLEVP 
% (Nonlinear Eigenvalue Problems) collection.
%
% The script computes the relative structured distance to singularity using
% both the left and right kernels of the associated polynomial matrix 
% (and takes the minimum of the two). This allows for a faster evaluation. For details, 
% see for example Section 6 of the Riemann-Oracle paper (arxiv.org/abs/2407.03957).

% Note: This script requires that the Riemann-Oracle method (https://github.com/fph/RiemannOracle)
% is available in the MATLAB path or is located in the current folder.
% This script also requires that the NLEVP collection to be available 
% in the MATLAB path or in a subfolder named 'nlevp'.

clear all;
rng(1);

[coeffs, ~] = nlevp('mobile_manipulator');
A_poly = cat(3, full(coeffs{1}), full(coeffs{2}), full(coeffs{3}));
A_poly_left = cat(3, full(coeffs{1}).', full(coeffs{2}).', full(coeffs{3}).');

norm_A = norm(A_poly(:));
d = floor(2 * (size(A_poly, 1) - 1) / 2);

fprintf('Evaluating SVS...\n');
tic; dist_right = run_experiment_SVS(A_poly, d); t_right = toc;
tic; dist_left = run_experiment_SVS(A_poly_left, d); t_left = toc;
dist_SVS = min(dist_right, dist_left) / norm_A;
fprintf('SVS Computed distance: %.8e. Time: %.2f s\n\n', dist_SVS, t_right + t_left);

fprintf('Evaluating Riemann-Oracle...\n');
tic; dist_right_RO = run_experiment_RO(A_poly, d); t_right_RO = toc;
tic; dist_left_RO = run_experiment_RO(A_poly_left, d); t_left_RO = toc;
dist_RO = min(dist_right_RO, dist_left_RO) / norm_A;
fprintf('Riemann-Oracle Computed distance: %.8e. Time: %.2f s\n', dist_RO, t_right_RO + t_left_RO);

function dist_SVS = run_experiment_SVS(A_poly, d)
    [Td_A, P_basis] = construct_basis(A_poly, d);
    Delta = structured_distance_to_singularity(Td_A, P_basis);
    dist_SVS = norm(Delta, 'fro') / sqrt(d+1);
end

function dist_RO = run_experiment_RO(A_poly, d)
    [Td_A, P_basis] = construct_basis(A_poly, d);
    problem = nearest_singular_structured_dense(P_basis, Td_A, true);
    opts = struct('verbosity', 0, 'y', 0, 'stopping_criterion', 1e-14);
    
    [x, ~, info] = penalty_method(problem, [], opts);
    regproblem = apply_regularization(problem, info.last_epsilon, info.y);
    Delta = regproblem.minimizer(x, struct());
    
    dist_RO = norm(Delta, 'fro') / sqrt(d+1);
end

function [Td_A, P_basis] = construct_basis(A_poly, d)
    idx = find(A_poly ~= 0);
    Td_A = polytoep(A_poly, d);
    P_basis = zeros(size(Td_A, 1), size(Td_A, 2), length(idx));
    
    for i = 1:length(idx)
        P_temp = zeros(size(A_poly));
        P_temp(idx(i)) = 1;
        P_basis(:, :, i) = polytoep(P_temp, d);
    end
end
