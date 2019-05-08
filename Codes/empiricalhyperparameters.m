function [hyperparameters] = empiricalhyperparameters(x,z,c,p,q,gprior)
%empiricalhyperparameters: defines empirical hyperparameters based on the
%data
% INPUTs:
%   x: covariate matrix of size nx(p+q)
%   z: observed response of size nxd 
%   c: censoring indicators of size nxb - for age at event variables
%   gprior: the constant g used in the gprior formulation
% OUTPUTs:
%    hyperparameter: is structure containing the specified empirical values
%    of the hyperparamters (beta0,U_iC,Sigma0,nu,mu0,ic,a1,a2,M)

n = size(x,1); %sample size
d = size(z,2); % dimension of response
b = size(c,2); % number of age at event variables

% Call inverse link function to obtain bounds on latent y
[lower,upper] = invlinkfunctions(z,c,x);

% Initialise latent y based on bounds
y=zeros(n,d);
% If both bounds are finite initialise at mid point
y(isfinite(upper)&isfinite(lower))=(upper(isfinite(upper)&isfinite(lower))+lower(isfinite(upper)&isfinite(lower)))/2;
for id=1:d
    if id<=b
        % For age at event variables: if lower finite and upper is Inf,
        % initialise at log(exp(lower)+1) (note: lower+1 can be an extreme
        % initialisation on the exponential scale).
        y(isinf(upper(:,id))&isfinite(lower(:,id)),id)=log( exp(lower(isinf(upper(:,id))&isfinite(lower(:,id)),id))+1);
        %if lower Inf and upper is finite, initialise at log(exp(upper)-1) 
        %(note: upper-1 can be an extreme initialisation on the exponential scale).
        y(isfinite(upper(:,id))&isinf(lower(:,id)),id)=log(exp(upper(isfinite(upper(:,id))&isinf(lower(:,id)),id))-1);
    else
        % For binary variables, if lower finite and upper is Inf,
        % initialise at lower+1
        y(isinf(upper(:,id))&isfinite(lower(:,id)),id)=lower(isinf(upper(:,id))&isfinite(lower(:,id)),id)+1;
        % If lower is Inf and upper is finite, initialise at upper-1
        y(isfinite(upper(:,id))&isinf(lower(:,id)),id)=upper(isfinite(upper(:,id))&isinf(lower(:,id)),id)-1;
    end
end

% Multivariate regression for estimates of coefficients and covariance
% matrix: beta_lm and Sigma_lm
X = [ones(n,1),x];
[beta_lm,Sigma_lm] = mvregress(X,y);

%% Set prior parameters for local linear multivariate regressions components
% beta | Sigma is Multivariate Gaussian with 
%    mean: beta0
%    covariance: kron(Sigma, U_iC)
% Sigma is Inverse Wishart with parameter Sigma0 and nu degrees of freedom

% Center Wishart prior on empircal Sigma_lm with nu degress of freedom
Sigma0=2*Sigma_lm;
nu=d+3;

% Center Gaussian prior on empirical beta_lm and use a gprior to specify
% the covariance matrix U
beta0=beta_lm;
U_iC=gprior*inv(X'*X)/(min(diag(Sigma0))/(nu-d-1));


%% Set prior parameters for covariate dependent weights
% Prior for parameters of continuous covariate is Normal-Gamma
% mu|tau is Normal with mean mu0 and variance (ic*tau)^{-1}
% tau is Gamma with parameters a1 and a2
mu0 = zeros(1,p);
a1 = 2*ones(1,p);
a2 = ones(1,p);
ic = 1/2*ones(1,p);

for i = 1:p
    %Fix mu0 at empirical mean
    mu0(i) = mean(x(:,i));
    % Fix a2 based on emprical range of data
    a2(i)=ic(i)*((max(x(:,i))-min(x(:,i)))/4)^2;
end

% Prior for parameters of binary expanded categorical covariates is Beta
% rho is Beta(alpha_rho(1), alpha_rho(2))
alpha_rho = ones(2,q);

%Prior parameter for DP/stick-breaking prior, i.e. stick-breaking
%proportions v_j iid from Beta(1,M)
M = 1;

hyperparameters = struct('beta0',beta0,'U_iC',U_iC,'Sigma0',Sigma0,'nu',nu,'mu0',mu0,'ic',ic,'a1',a1,'a2',a2,'alpha_rho',alpha_rho,'M',M);

end

