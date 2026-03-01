function [Delta,inner_iteration_count] = sparse_distance_to_singularity(A,augmented_lagrangian)

A = sparse(A);
[n,m] = size(A);

if not(exist('augmented_lagrangian', 'var'))
    augmented_lagrangian = 0;
end

saved_warning_state = warning();
warning('off', 'MATLAB:nearlySingularMatrix'); %This inevitably happens when epsilon --> 0.

[row,col] = find(A);

eps_range = 10.^(-(1:0.5:17));

% Initial guess for the vector in the kernel
V = orth(randn(n,1));

inner_iteration_count = 0;

if augmented_lagrangian; eps_range = 0.5.^(0:100); end
y=zeros(n,1);

eps_reg = 1e-1;
while eps_reg > 1e-17
    reg_norms = [inf];
    index = 0;
    while index <= 100
        index = index+1;

        M = sparse(row,1:length(row),V(col),n,length(row));
        delta = - M'*((1./(vecnorm(M,2,2).^2+eps_reg*ones(n,1))).*(A*V + eps_reg*y));
        Delta = sparse(row,col,delta,n,n);

        reg_norms = [reg_norms sqrt(norm(Delta,'f')^2+1/eps_reg*norm((A+Delta)*V + eps_reg*y)^2)];

        if abs(reg_norms(end-1)-reg_norms(end)) < 1e-8
            break
        end
        if reg_norms(end-1) < reg_norms(end) - 1e-8
            % Value of objective function increased between iterations.
            % Can be commented out. Keeping this for now for debugging
            % purposes.
            % keyboard
        end

        if index > 100
            break
        end
              
        if augmented_lagrangian && ~all(y == 0)
            AplusD = A+Delta;

            problem.cost = @(V) 0.5*norm((AplusD)*V + eps_reg*y,'f')^2;
            problem.egrad = @(V) (AplusD'*(AplusD*V) + AplusD'*eps_reg*y);
            problem.ehess = @(V,dV) AplusD'*(AplusD*dV);

            problem.M = spherecomplexfactory(m);
            options.tolgradnorm = 5e-6;
            options.verbosity = 0; 
            options.maxiter = 1;
            options.maxinner = 10; 

            [x, ~, ~] = trustregions(problem, V, options);
            V = x;
        else
            AplusD = A+Delta;
            if eps_reg <= 1e-6
                [U,S,W] = svds(AplusD,1,'smallest');
            else
                if nnz(AplusD)/numel(AplusD) > 0.3; 
                    ATA=full(AplusD)'*full(AplusD); 
                else
                    ATA=AplusD'*AplusD;
                end
                if nnz(ATA)/numel(ATA) > 0.4; ATA = full(ATA); end
                [Rchol, flag] = chol(ATA);
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

    % Flexible update
    cost = norm(delta,'f')^2 + (1/eps_reg)*norm((A+Delta)*V+eps_reg*y)^2;
    eps_reg_new = eps_reg*1e-2;
    cost_new = norm(delta,'f')^2 + (1/eps_reg_new)*norm((A+Delta)*V+eps_reg_new*y)^2;
    while cost_new > 2.5 * cost                     
        eps_reg_new = eps_reg_new * 1.1;
        cost_new = norm(delta,'f')^2 + (1/eps_reg_new)*norm((A+Delta)*V+eps_reg_new*y)^2;
        if eps_reg_new > 0.95*eps_reg
            break
        end
    end
    eps_reg = eps_reg_new;

    % Standard update
    % eps_reg = eps_reg/sqrt(10);

    inner_iteration_count = inner_iteration_count + length(reg_norms) - 1; 

end

warning(saved_warning_state);

end