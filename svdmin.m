function [u s v]=svdmin(A, Rchol);
% code by Ethan N. Epperly, Yuji Nakatsukasa and Taejun Park

m = size(A,1); n = size(A,2);
blksize = 1;
maxiter = 20;
tol = 1e-13; %can be changed
V0 = Rchol\randn(n,blksize);
ress = []; respre = inf;

% initialization
%[C,T] = Rayleigh_Ritz(A*V0,V0);

[~,RR] = qr(V0,0);
temp = (A*V0)/RR;
[~,T,Z] = svd(temp,'econ');
C = (RR\Z);
T = diag(T);

X = V0*C; % solution vector
AX = A*X;
R = A'*(AX)-X*diag(T.^2);
P = []; % search direction
AP = [];

for i = 1:maxiter
    %W = Prec * (Prec' * R);
    W = Rchol\((Rchol')\R);
    W = W - [X,P]*([X,P]'*W);
    [W,~] = qr(W - [X,P]*([X,P]'*W),0); % orthogonalize against [X,P]

    S = [X,P,W];
    AS = [AX,AP,A*W];

    %[C,T] = Rayleigh_Ritz(AS,S);

    [~,RR] = qr(S,0);
    temp = AS/RR;
    [~,T,Z] = svd(temp,'econ');
    C = (RR\Z);
    T = diag(T);


    [T,IX] = sort(T); T = T(1:blksize);
    C = C(:,IX); % sort eigenvalues in nondecreasing order

    [QC,~] = qr(C(1:blksize,blksize+1:end)',0);
    X = S * C(:,1:blksize);
    AX = AS * C(:,1:blksize);
    R = A'*(AX) - X*diag(T.^2);
    P = S * C(:,blksize+1:end) * QC;
    AP = AS * C(:,blksize+1:end) * QC;

    % 1 matvec with Prec, Prec', A, A' per iteration

    %if ~isempty(summary); stats(end+1,:) = summary(X,T); end
    % add stopping criteria later - involve gap?
    u = A*X;
    ress = [ress norm(u)];
    if abs(ress(end)-respre)<tol
        % disp(['converged in ',num2str(i),' LOBPCG steps'])
        break
    end
    respre = ress(end);
end
v=X;
s=respre;
u=u/s;