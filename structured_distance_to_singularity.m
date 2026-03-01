function [Delta,inner_iteration_count] = structured_distance_to_singularity(A,P,augmented_lagrangian,regularize,alternate,use_left_kernel)
[n,m] = size(A);

if not(exist('augmented_lagrangian', 'var'))
    augmented_lagrangian = 0;
end
if not(exist('regularize', 'var'))
    regularize = 1;
end
if not(exist('alternate', 'var'))
    alternate = 0;
end
if not(exist('use_left_kernel', 'var'))
    use_left_kernel = 0;
end

assert(alternate <=2 && 0 <= alternate)
% Augmented lagrangian is not compatible with the alternating approach
assert(~(alternate && augmented_lagrangian))
% Augmented lagrangian is not compatible with the nonregularised approach
assert(~(~regularize && augmented_lagrangian))

saved_warning_state = warning();
warning('off', 'MATLAB:nearlySingularMatrix'); %This inevitably happens when epsilon --> 0.

P_mat = matricize(P, 3);
P_mat = sparse(P_mat);

P_mat2 = matricize(P, 2);
P_mat2 = sparse(P_mat2).';

if use_left_kernel || ~regularize || alternate == 1
    P_t = pagetranspose(P);
    
    Pt_mat2 = matricize(P_t, 2);
    Pt_mat2 = sparse(Pt_mat2).';
end

eps_range = 10.^(-(1:0.5:17));

% Initial guess for the vector in the kernel
V0 = orth(randn(m));
V = V0(:,1); 
U=V;
% Norm: 1.288486e-02, sigma_min: 1.952500e-17

inner_iteration_count = 0;

if augmented_lagrangian; eps_range = 0.5.^(0:100); end
y=zeros(n,1);
for eps_reg = eps_range
    norms = [inf];
    reg_norms = [inf];
    count=0;
    index = 0;
    while index <= 100
        index = index+1;
        if regularize
            if use_left_kernel
                U1 = U(:,end)';
                M = make_M2(Pt_mat2, size(P,1), size(P,3), U1.');
                delta = - M'*((M*M' + eye(n)*eps_reg)\(U1*A).');
                Delta = sparse(delta.' * P_mat);
                Delta = tensorize(Delta, 3, [n,n]);
                reg_norms = [reg_norms sqrt(norm(Delta,'f')^2+1/eps_reg*norm(U1*(A+Delta),'f')^2)];
            else
                M = make_M2(P_mat2, size(P,1), size(P,3), V);
                delta = - M'*((M*M' + eye(n)*eps_reg)\(A*V + eps_reg*y)); 
                Delta = sparse(delta.' * P_mat);
                Delta = tensorize(Delta, 3, [n,m]);
                reg_norms = [reg_norms sqrt(norm(Delta,'f')^2+1/eps_reg*norm((A+Delta)*V + eps_reg*y)^2)];
            end
            if abs(reg_norms(end-1)-reg_norms(end)) < 1e-8
                break
            end
            if reg_norms(end-1) < reg_norms(end) - 1e-8
                % Value of objective function increased between iterations.
                % Can be commented out. Keeping this for now for debugging
                % purposes.
                % keyboard
            end
        else
            M = make_M2(P_mat2, size(P,1), size(P,3), V);
            delta = - pinv(M)*A*V;
            Delta = sparse(delta.' * P_mat);
            Delta = tensorize(Delta, 3, [n,m]);
            norms = [norms norm(Delta,'f')];
        end

        if index > 100
            break
        end
        
        if ~regularize
            index = index+1;
            AplusD = A+Delta;
            [U,S,W] = svd(AplusD);
            U1 = U(:,end)';

            M = make_M(P_t, U1.');
            delta = - pinv(M)*(U1(1,:)*A).';
            Delta = sparse(delta.' * P_mat);
            Delta = tensorize(Delta, 3, [n,n]);
        
            norms = [norms norm(Delta,'f')];
         
            if abs(norms(end-1)-norms(end)) < 1e-8
                break
            end
        else
            switch alternate

                case 1
                    index = index+1;
                    AplusD = A+Delta;
                    if eps_reg <= 1e-6 || regularize == 0
                        [U,S,W] = svds(AplusD,1,'smallest');
                    else
                        [Rchol, flag] = chol(AplusD'*AplusD);
                        if flag ~= 0
                            [U,S,W] = svds(AplusD,1,'smallest');
                        else
                            [U,S,W] = svdmin(AplusD, Rchol);
                        end
                    end
                    U1 = U';
                    M = make_M2(Pt_mat2, size(P,1), size(P,3), U1.');
                    delta = - M'*((M*M' + eye(n)*eps_reg)\(U1*A).');
                    Delta = sparse(delta.' * P_mat);
                    Delta = tensorize(Delta, 3, [n,n]);
                    reg_norms = [reg_norms sqrt(norm(Delta,'f')^2+1/eps_reg*norm(U1*(A+Delta),'f')^2)];
                case 2
                    index = index+1;
                    [U,S,W] = svd(full(A+Delta));
                    U(:,[end-1 end]) = U(:,[end end-1]);
                    W(:,[end-1 end]) = W(:,[end end-1]);
                    S(end-1:end,end-1:end) = [0 1; 1 0]*S(end-1:end,end-1:end)*[0 1; 1 0];

                    M = reshape(pagemtimes(pagemtimes(U(:,1:n-1)', P),W(:,n-1:n)), [2*(n-1) size(P,3)]);
                    MtimesA = U(:,1:n-1)'*A*W(:,n-1:n);
                    delta = - M'*((M*M' + eye(2*(n-1))*eps_reg)\MtimesA(:));
        
                    Delta = sparse(delta.' * P_mat);
                    Delta = tensorize(Delta, 3, [n,n]);

                    constraint = M*delta + MtimesA(:);

                    reg_norms = [reg_norms sqrt(norm(Delta,'f')^2+1/eps_reg*norm(constraint,'f')^2)];
                
                    norms = [norms norm(Delta,'f')];
                 
            end

            if abs(reg_norms(end-1)-reg_norms(end)) < 1e-8
                break
            end

        end

        
        if augmented_lagrangian && ~all(y == 0)
            AplusD = A+Delta;
            ATA = AplusD'*AplusD;

            problem.cost = @(V) 0.5*norm((AplusD)*V + eps_reg*y,'f')^2;
            problem.egrad = @(V) (ATA*V + AplusD'*eps_reg*y);
            problem.ehess = @(V,dV) ATA*dV;

            problem.M = spherecomplexfactory(m);
            options.tolgradnorm = 5e-6;
            options.verbosity = 0; 
            options.maxiter = 1;
            options.maxinner = 100; 

            [x, ~, ~] = trustregions(problem, V, options);
            V = x;
        else
            AplusD = A+Delta;
            if eps_reg <= 1e-6 || regularize == 0
                [U,S,W] = svds(AplusD,1,'smallest');
            else
                [Rchol, flag] = chol(AplusD'*AplusD);
                if flag ~= 0
                    [U,S,W] = svds(AplusD,1,'smallest');
                else
                    [U,S,W] = svdmin(AplusD, Rchol);
                end
            end
            V = W(:,end);
        end

    end
    
   if augmented_lagrangian
        cons = (A+Delta)*V;
        constraint_satisfied = norm(cons,'f') < 1e-14;
        if constraint_satisfied
            break
        end
        y = y + 1/eps_reg * cons;
    end
    
    inner_iteration_count = inner_iteration_count + length(reg_norms) - 1; 

end

if ~issparse(A); Delta = full(Delta); end

warning(saved_warning_state);

function [c,store] = cost(V,AplusD,eps_reg,y,store)
    if ~isfield(store, 'c')
        store.c = norm((AplusD)*V + eps_reg*y,'f')^2;
    end
    c = store.c;
end

function [g,store] = egrad(V,AplusD,ATA,eps_reg,y,store)
    if ~isfield(store, 'g')
        store.g = 2*ATA*V + 2*AplusD'*eps_reg*y;
    end
    g = store.g;
end

function [h,store] = ehess(dV,ATA,store)
    store.h = 2*ATA*dV;
    h = store.h;
end

function M = make_M(P, v)
    M = reshape(pagemtimes(P, v), [size(P,1) size(P,3)]);
end

function M = make_M2(P, n, k, v)
    M = reshape(P*v, [n k]);
end


end

