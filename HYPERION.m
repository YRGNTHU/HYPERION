function [Signatures,T,time] = HYPERION(Y,N,lambda)
t0 = clock;
%% PCA
[l n]= size(Y);
d= mean(Y,2);
U= Y-d*ones(1,n);
R= U*U';
[eV,~]= eig(R);
C= eV(:,l-N+2:end);
Yd= C'*(Y-d*ones(1,n)); 
%% HYPERION
[F,c]= LJE(N,Yd);
CoX= inv(F)*(Yd-c*ones(1,n));
[S,T,U] = EstPure(CoX,N,lambda);
Signatures= C*(F*S+ c*ones(1,N))+ d*ones(1,N);
time = etime(clock,t0);

%% subprogram 1
function [E,c]= LJE(p,Y)
[b,h,~] = vert2con_with_index(Y'); 
[m,~]= size(b);
n = p-1;
cvx_begin quiet
variable E(n,n) symmetric
variable c(n)
maximize( det_rootn( E ) )
subject to
for j = 1:m
    norm( E*b(j,:)', 2 ) + b(j,:)*c <= h(j);
end
cvx_end

%% subprogram 2
function [S,T,U] = EstPure(CoX,p,eta)
[S,~]= SPA(CoX,p);
T=(1/p)*ones(p,size(CoX,2));
alpha= 1*((1/(factorial(p-1)))*(p^(0.5*p))*((p-1)^(0.5*(p-1))))^(1/(p-1));
S0= eye(p)-(1/p)*ones(p,p);
[M0,~]= eig(S0*S0'); M0(:,1)=[]; M0= (((factorial(p-1))/(sqrt(p)))^(1/(p-1)))*(M0');
[D2,~,D1]= svd(alpha*S*M0'); U= D1*D2';
j=0;
while j<200, 
    S= (CoX*(T')+(eta*alpha*(U')*M0))*inv(T*T'+(eta*eye(p))); [D2,~,D1]= svd(alpha*S*M0');
    T = sunsal([S;ones(1,p)],[CoX;ones(1,size(CoX,2))],'POSITIVITY','yes','ADDONE','yes','lambda',0,'AL_ITERS',100,'TOL',1e-6,'X0',T);
    U= D1*D2';
    j=j+1;
end
return;

%% subprogram 3
function [A,b,index] = vert2con_with_index(V)
% VERT2CON - convert a set of points to the set of inequality constraints
%            which most tightly contain the points; i.e., create
%            constraints to bound the convex hull of the given points
%
% [A,b,index] = vert2con_with_index(V)
%
% V = a set of points, each ROW of which is one point
% A,b = a set of constraints such that A*x <= b defines
%       the region of space enclosing the convex hull of
%       the given points
%
% For n dimensions:
% V = p x n matrix (p vertices, n dimensions)
% A = m x n matrix (m constraints, n dimensions)
% b = m x 1 vector (m constraints)
%
% NOTES: (1) In higher dimensions, duplicate constraints can
%            appear. This program detects duplicates at up to 6
%            digits of precision, then returns the unique constraints.
%        (2) See companion function CON2VERT.
%        (3) ver 1.0: initial version, June 2005.
%        (4) ver 1.1: enhanced redundancy checks, July 2005
%        (5) Written by Michael Kleder
% 
% EXAMPLE:
%
% V=rand(20,2)*6-2;
% [A,b]=vert2con(V)
% figure('renderer','zbuffer')
% hold on
% plot(V(:,1),V(:,2),'r.')
% [x,y]=ndgrid(-3:.01:5);
% p=[x(:) y(:)]';
% p=(A*p <= repmat(b,[1 length(p)]));
% p = double(all(p));
% p=reshape(p,size(x));
% h=pcolor(x,y,p);
% set(h,'edgecolor','none')
% set(h,'zdata',get(h,'zdata')-1) % keep in back
% axis equal
% set(gca,'color','none')
% title('A*x <= b  (1=True, 0=False)')
% colorbar

k = convhulln(V);
temp = vec(k); index = union(temp,temp(1));
c = mean(V(unique(k),:));
V=V-repmat(c,[size(V,1) 1]);
A  = NaN*zeros(size(k,1),size(V,2));
rc=0;
for ix = 1:size(k,1)
    F = V(k(ix,:),:);
    if rank(F,1e-5) == size(F,1)
        rc=rc+1;
        A(rc,:)=F\ones(size(F,1),1);
    end
end
A=A(1:rc,:);
b=ones(size(A,1),1);
b=b+A*c';
[null,I]=unique(num2str([A b],6),'rows');
A=A(I,:);
b=b(I);
return

%% subprogram 4
function [z,res_p,res_d] = sunsal(M,y,varargin)

%% --------------- Description --------------------------------------------
%
%  SUNSAL solves the following l2-l1 optimization  problem
%  [size(M) = (L,p); size(X) = (p,N)]; size(Y) = (L,N)]
%
%         min  (1/2) ||M X-y||^2_F + lambda ||X||_1
%          X
%
%  where ||X||_1 = sum(sum(abs(X)).
%
%    CONSTRAINTS ACCEPTED:
%
%    1) POSITIVITY:  X >= 0;
%    2) ADDONE:  sum(X) = ones(1,N);
%
%    NOTES:
%       1) The optimization w.r.t each column of X is decoupled. Thus,
%          SUNSAL solves N simultaneous problems.
%
%       2) SUNSAL solves the following  problems:
%
%          a) BPDN - Basis pursuit denoising l2-l1
%                    (lambda > 0, POSITIVITY = 'no', ADDONE, 'no')
%
%          b) CBPDN - Constrained basis pursuit denoising l2-l1
%                    (lambda > 0, POSITIVITY = 'yes', ADDONE, 'no')
%
%          c) CLS   - Constrained least squares
%                     (lambda = 0, POSITIVITY = 'yes', ADDONE, 'no')
%
%          c) FCLS   - Fully constrained least squares
%                     (lambda >=0 , POSITIVITY = 'yes', ADDONE, 'yes')
%                      In this case, the regularizer ||X||_1  plays no role,
%                      as it is constant.
%
%
%% -------------------- Line of Attack  -----------------------------------
%
%  SUNSAL solves the above optimization problem by introducing a variable
%  splitting and then solving the resulting constrained optimization with
%  the augmented Lagrangian method of multipliers (ADMM).
%
%
%         min  (1/2) ||M X-y||^2_F + lambda ||Z||_1
%          X,Z
%         subject to: sum(X) = ones(1,N)); Z >= 0; X = Z
%
%  Augmented Lagrangian (scaled version):
%
%       L(X,Z,D) = (1/2) ||M X-y||^2_F + lambda ||Z||_1 + mu/2||X-Z-D||^2_F
%
%  where D are the scale Lagrange multipliers
%
%
%  ADMM:
%
%      do
%        X  <-- arg min L(X,Z,D)
%                    X, s.t: sum(X) = ones(1,N));
%        Z  <-- arg min L(X,Z,D)
%                    Z, s.t: Z >= 0;
%        D  <-- D - (X-Z);
%      while ~stop_rulde
%
%For details see
%
%
% [1] J. Bioucas-Dias and M. Figueiredo, “Alternating direction algorithms
% for constrained sparse regression: Application to hyperspectral unmixing?,
% in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal
% Processing-WHISPERS'2010, Raykjavik, Iceland, 2010.
%
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
%  M - [L(channels) x p(endmembers)] mixing matrix
%
%  y - matrix with  L(channels) x N(pixels).
%      each pixel is a linear mixture of p endmembers
%      signatures y = M*x + noise,
%
%
%
%
%%  ====================== Optional inputs =============================
%
%  'AL_ITERS' - Minimum number of augmented Lagrangian iterations
%               Default: 100;
%
%  lambda - regularization parameter. lambda is either a scalar
%           or a vector with N components (one per column of x)
%           Default: 0.
%
%
%  'POSITIVITY'  = {'yes', 'no'}; Enforces the positivity constraint:
%                   X >= 0
%                   Default 'no'
%
%  'ADDONE'  = {'yes', 'no'}; Enforces the positivity constraint: X >= 0
%              Default 'no'
%
%   'TOL'    - tolerance for the primal and  dual residuals
%              Default = 1e-4;
%
%
%  'verbose'   = {'yes', 'no'};
%                 'no' - work silently
%                 'yes' - display warnings
%                  Default 'no'
%
%%  =========================== Outputs ==================================
%
% X  =  [pxN] estimated mixing matrix
%
%

%%
% ------------------------------------------------------------------
% Author: Jose Bioucas-Dias, 2009
%
%

%
%% -------------------------------------------------------------------------
%
% Copyright (July, 2009):        Jos? Bioucas-Dias (bioucas@lx.it.pt)
%
% SUNSAL is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------



%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% mixing matrixsize
[LM,p] = size(M);
% data set size
[L,N] = size(y);
if (LM ~= L)
    error('mixing matrix M and data set y are inconsistent');
end


%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
AL_iters = 1000;
lambda = 0.0;
verbose = 'off';
positivity = 'no';
addone = 'no';
tol = 1e-4; 
x0 = 0;

%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'LAMBDA'
                lambda = varargin{i+1};
                if (sum(sum(lambda < 0)) >  0 )
                    error('lambda must be positive');
                end
            case 'POSITIVITY'
                positivity =  varargin{i+1};
            case 'ADDONE'
                addone = varargin{i+1};
            case 'TOL'
                tol = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= p) | (size(x0,2) ~= N)
                    error('initial X is  inconsistent with M or Y');
                end
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

%---------------------------------------------
%  If lambda is scalar convert it into vector
%---------------------------------------------
Nlambda = size(lambda);
if Nlambda == 1
    lambda = lambda*ones(p,N);
elseif Nlambda ~= N
    error('Lambda size is inconsistent with the size of the data set');
else
    lambda = repmat(lambda(:)',p,1);
end


norm_y = sqrt(mean(mean(y.^2)));
M = M/norm_y;
y = y/norm_y;
lambda = lambda/norm_y^2;



%%
%---------------------------------------------
% just least squares
%---------------------------------------------
if sum(sum(lambda == 0)) &&  strcmp(positivity,'no') && strcmp(addone,'no')
    z = pinv(M)*y;
    res_p = 0; 
    res_d = 0;
    return
end
%---------------------------------------------
% least squares constrained (sum(x) = 1)
%---------------------------------------------
SMALL = 1e-12;
B = ones(1,p);
a = ones(1,N);

if  strcmp(addone,'yes') && strcmp(positivity,'no')
    F = M'*M;
    
    % test if F is invertible
    if rcond(F) > SMALL
        IF = inv(F);
        z = IF*M'*y-IF*B'*inv(B*IF*B')*(B*IF*M'*y-a);
        res_p = 0;
        res_d = 0;
        return
    end
end


%%
%---------------------------------------------
%  Constants and initializations
%---------------------------------------------
mu_AL = 0.01;
mu = 10*mean(lambda(:)) + mu_AL;
[UF,SF] = svd(M'*M);
sF = diag(SF);
IF = UF*diag(1./(sF+mu))*UF';
Aux = IF*B'*inv(B*IF*B');
x_aux = Aux*a;
IF1 = (IF-Aux*B*IF);
yy = M'*y;

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
if x0 == 0
    x= IF*M'*y;
else
    x = x0;
end

z = x;
% scaled Lagrange Multipliers
d  = 0*z;


%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol1 = sqrt(N*p)*tol;
tol2 = sqrt(N*p)*tol;
i=1;
res_p = inf;
res_d = inf;
maskz = ones(size(z));
mu_changed = 0;

if sum(lambda(:) > 0)
    % implement soft_th
    while (i <= AL_iters) && ((abs (res_p) > tol1) || (abs (res_d) > tol2))
        
        % save z to be used later
        if mod(i,10) == 1
            z0 = z;
        end
        
        % minimize with respect to z
        z =  soft(x-d,lambda/mu);
        
        % teste for positivity
        if strcmp(positivity,'yes')   
            z = max(z,0);
        end
        
        % teste for sum-to-one
        if strcmp(addone,'yes')
            x = IF1*(yy+mu*(z+d))+x_aux;
        else
            x = IF*(yy+mu*(z+d));
        end
        d = d -(x-z);
        
        % update mu so to keep primal and dual residuals whithin a factor of 10
        if mod(i,10) == 1
            res_p = norm(x-z,'fro');
            res_d = mu*norm(z-z0,'fro');
            if  strcmp(verbose,'yes')
                fprintf(' i = %f, res_p = %f, res_d = %f\n',i,res_p,res_d)
            end
            
            % update mu
            if res_p > 10*res_d
                mu = mu*2;
                d = d/2;
                mu_changed = 1;
            elseif res_d > 10*res_p
                mu = mu/2;
                d = d*2;
                mu_changed = 1;
            end
            if  mu_changed
                % update IF and IF1
                
                IF = UF*diag(1./(sF+mu))*UF';
                Aux = IF*B'*inv(B*IF*B');
                x_aux = Aux*a;
                IF1 = (IF-Aux*B*IF);
                mu_changed = 0;
            end
        end
        i=i+1;
    end
else
    while (i <= AL_iters) && ((abs (res_p) > tol1) || (abs (res_d) > tol2))
        % save z to be used later
        if mod(i,10) == 1
            z0 = z;
        end
        z=x-d;
        
        % teste for positivity
        if strcmp(positivity,'yes')
            z = max(z,0);
        end
        
        % teste for sum-to-one
        if strcmp(addone,'yes')
            x = IF1*(yy+mu*(z+d))+x_aux;
        else
            x = IF*(yy+mu*(z+d));
        end
        
        % Lagrange multipliers update
        d = d -(x-z);
        
        % update mu so to keep primal and dual residuals whithin a factor of 10
        if mod(i,10) == 1
            res_p = norm(x-z,'fro');
            res_d = mu*norm(z-z0,'fro');
            if  strcmp(verbose,'yes')
                fprintf(' i = %f, res_p = %f, res_d = %f\n',i,res_p,res_d)
            end
            
            % update mu 
            if res_p > 10*res_d
                mu = mu*2;
                d = d/2;
                mu_changed = 1;
            elseif res_d > 10*res_p
                mu = mu/2;
                d = d*2;
                mu_changed = 1;
            end
            if  mu_changed
                % update IF and IF1 
            
                IF = UF*diag(1./(sF+mu))*UF';
                Aux = IF*B'*inv(B*IF*B');
                x_aux = Aux*a;
                IF1 = (IF-Aux*B*IF);
                mu_changed = 0;
            end
        end
        i=i+1;
    end
end

%% subprogram 5
function [A_est, time] = SPA(X,N)
t0 = clock;

%------------------------ Step 1 ------------------------
[M L ] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
[eV D] = eig(U*U');
C = eV(:,M-N+2:end);
Xd = C'*(X-d*ones(1,L));

%------------------------ Step 2 ------------------------
alpha_tilde = SPAinHyperCSI(Xd,L,N);

%------------------------ Step 3 ------------------------
A_est = C * alpha_tilde + d * ones(1,N);
time = etime(clock,t0);

%% subprogram 6
function [bi] = compute_bi(a0,i,N)
Hindx = setdiff([1:N],[i]);
A_Hindx = a0(:,Hindx);
A_tilde_i = A_Hindx(:,1:N-2)-A_Hindx(:,N-1)*ones(1,N-2);
bi = A_Hindx(:,N-1)-a0(:,i);
bi = (eye(N-1) - A_tilde_i*(pinv(A_tilde_i'*A_tilde_i))*A_tilde_i')*bi;
bi = bi/norm(bi);
return;

%% subprogram 7
function [alpha_tilde] = SPAinHyperCSI(Xd,L,N)

% Reference:
% [1] W.-K. Ma, J. M. Bioucas-Dias, T.-H. Chan, N. Gillis, P. Gader, A. J. Plaza, A. Ambikapathi, and C.-Y. Chi, 
% ``A signal processing perspective on hyperspectral unmixing,¡¨ 
% IEEE Signal Process. Mag., vol. 31, no. 1, pp. 67¡V81, 2014.
% 
% [2] S. Arora, R. Ge, Y. Halpern, D. Mimno, A. Moitra, D. Sontag, Y. Wu, and M. Zhu, 
% ``A practical algorithm for topic modeling with provable guarantees,¡¨ 
% arXiv preprint arXiv:1212.4777, 2012.
%======================================================================
% An implementation of successive projection algorithm (SPA) 
% [alpha_tilde] = SPA(Xd,L,N)
%======================================================================
%  Input
%  Xd is dimension-reduced (DR) data matrix.
%  L is the number of pixels.   
%  N is the number of endmembers.
%----------------------------------------------------------------------
%  Output
%  alpha_tilde is an (N-1)-by-N matrix whose columns are DR purest pixels.
%======================================================================
%======================================================================

%----------- Define default parameters------------------
con_tol = 1e-8;
num_SPA_itr = N;
N_max = N;

%------------------------ initialization of SPA ------------------------
A_set=[]; Xd_t = [Xd; ones(1,L)]; index = [];
[val ind] = max(sum( Xd_t.^2 ));
A_set = [A_set Xd_t(:,ind)];
index = [index ind];
for i=2:N
    XX = (eye(N_max) - A_set * pinv(A_set)) * Xd_t;
    [val ind] = max(sum( XX.^2 )); 
    A_set = [A_set Xd_t(:,ind)]; 
    index = [index ind]; 
end
alpha_tilde = Xd(:,index);

%------------------------ post-processing of SPA ------------------------
current_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
for jjj = 1:num_SPA_itr
    for i = 1:N
        b(:,i) = compute_bi(alpha_tilde,i,N);
        b(:,i) = -b(:,i);
        [const idx] = max(b(:,i)'*Xd);
        alpha_tilde(:,i) = Xd(:,idx);
    end
    new_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
    if (new_vol - current_vol)/current_vol  < con_tol
        break;
    end
end
return;