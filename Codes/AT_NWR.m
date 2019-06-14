function [particles, logweight, loglike_MCMC, loglike_SMC, ESS_SMC] = AT_NWR(z, c, x, p, q, mcmcsmc, hyperparameters)
%AT_NWR runs the adaptive truncation algorithm for the BNP regression model
% with normalized weights.
% INPUTS
% z: contains the response variables of size nxd, where the first b are 
%    ages at events and d-b are binary
% c: censoring dummies (nxb)
% x: covariate matrix of size nx(p+q), where the first p are numerical and
%    the last q are binary expansions of categorical variables (taking
%    categories 1 or 2!!))
% p: number of numerical covariates
% q: number of discrete (binary expanded) covariates
% mcmcsmc: contains parameters for the algorithm
% hyperparameters: contains the hyperparameters of the model
% OUTPUTS
% particles: structure of particles produced by the algorithm
% logweight: log of the unnormalised weights of the particles
% loglike_MCMC: loglikelihood saved at every iterations of the MCMC
% loglike_SMC: loglikelihood saved at for each particle and every
%              truncation level
% ESS_SMC: effective sample size at every truncation level/step of SMC

%% Initialize
n = size(x,1); % sample size
d = size(z,2); % dimension of response
b = size(c,2); % dimension of age at event response

% Algorithm parameters
start_trunc = mcmcsmc.start_trunc; % initial truncation level
numbofparts = mcmcsmc.numbofparts; % number of particles/ MCMC samples for initial MCMC
burnin = mcmcsmc.burnin; % burnin for initial MCMC
every = mcmcsmc.every; % thining factor for initial MCMC
numbofMCMC = mcmcsmc.numbofMCMC; % rejuvenation SMC: small number of MCMC iterations
top_trunc = mcmcsmc.top_trunc; % maximum truncation level
% SMC stopping rule: STOP if |ESS_{J+1}-ESS_J|<epsilon_trunc*numbofparts more than numb_trunc times
epsilon_trunc = mcmcsmc.epsilon_trunc;  
numb_trunc = mcmcsmc.numb_trunc;

% Total number of MCMC iterations
numbofits = burnin + numbofparts * every;

% Hyperparameters
beta0 = hyperparameters.beta0;
U_iC = hyperparameters.U_iC;
Sigma0 = hyperparameters.Sigma0;
nu = hyperparameters.nu;
mu0 = hyperparameters.mu0;
ic = hyperparameters.ic;
a1 = hyperparameters.a1;
a2 = hyperparameters.a2;
M = hyperparameters.M;
alpha_rho = hyperparameters.alpha_rho;

% Covariates with intercept
X = [ones(n,1),x];

% Latent variables Y and bounds
y_MCMC = zeros(n,d); % y values at each MCMC iteration
y = zeros(numbofparts,n,d); % saved y values across particles
% Call inverse link function to obtain bounds on latent y
[lower,upper] = invlinkfunctions(z,c,x);

% Initialize particles
V = ones(numbofparts, top_trunc);
mu = zeros(numbofparts, top_trunc, p);
tau = zeros(numbofparts, top_trunc, p);
beta_Vec = zeros(numbofparts, top_trunc, (p+q+1)*d);
beta = zeros(numbofparts, top_trunc, (p+q+1), d);
Sigma = zeros(numbofparts, top_trunc, d, d);
W = zeros(numbofparts, top_trunc); 
rho = zeros(q, numbofparts, top_trunc);


% Initialise matrices saving determinant and LDL decomposition of Sigma
det_Sigma = zeros(numbofparts, top_trunc); % determinant of Sigma
LD_Sigma_Vec = zeros(numbofparts, top_trunc, d*(d+1)/2); % lower triangular matrix of LDL decomposition of Sigma
diag_D = zeros(numbofparts, top_trunc, d); % diagonal of LDL decomposition of Sigma

% Initialise matrices saving likelihood
loglike_MCMC_save = zeros(1,numbofits); % loglikelihood during MCMC steps at each iteration
loglike = zeros(numbofparts, 1); % loglikelihood at each saved particle
loglike_SMC = zeros(numbofparts, top_trunc); % loglikelihood during SMC steps for each particle and truncation level
% Likelihood auxiliary calculations for each particle, sample, truncation level:
num = zeros(numbofparts,n,top_trunc); % multivariate linear regression components
unnorm = zeros(numbofparts,n,top_trunc); % unnormalised covariate depedent weights
norm_weights = zeros(numbofparts,n,top_trunc); % normalised covariate depedent weights

%Initial truncation level:
it = start_trunc - 1;
%Save ESS for SMC part
ESS_SMC = zeros(top_trunc - start_trunc,1);

%% Starting values for parameters in MCMC

% Initial value of latent y based on bounds
% If both bounds are finite initialise at mid point
y_MCMC(isfinite(upper)&isfinite(lower))=(upper(isfinite(upper)&isfinite(lower))+lower(isfinite(upper)&isfinite(lower)))/2;
for id=1:d
    if id<=b
        % For age at event variables: if lower finite and upper is Inf,
        % initialise at log(exp(lower)+1) (note: lower+1 can be an extreme
        % initialisation on the exponential scale).
        y_MCMC(isinf(upper(:,id))&isfinite(lower(:,id)),id)=log( exp(lower(isinf(upper(:,id))&isfinite(lower(:,id)),id))+1);
        %if lower Inf and upper is finite, initialise at log(exp(upper)-1) 
        %(note: upper-1 can be an extreme initialisation on the exponential scale).
        y_MCMC(isfinite(upper(:,id))&isinf(lower(:,id)),id)=log(exp(upper(isfinite(upper(:,id))&isinf(lower(:,id)),id))-1);
    else
        % For binary variables, if lower finite and upper is Inf,
        % initialise at lower+1
        y_MCMC(isinf(upper(:,id))&isfinite(lower(:,id)),id)=lower(isinf(upper(:,id))&isfinite(lower(:,id)),id)+1;
        % If lower is Inf and upper is finite, initialise at upper-1
        y_MCMC(isfinite(upper(:,id))&isinf(lower(:,id)),id)=upper(isfinite(upper(:,id))&isinf(lower(:,id)),id)-1;
    end
end

%Initial value of Sigma = prior mean across components
Sigma_MCMC = zeros(d, d, it);
det_Sigma_MCMC = zeros(1,it);
LD_Sigma_Vec_MCMC = zeros(d*(d+1)/2,it);
diag_D_MCMC = zeros(d,it);
for j = 1:it
    Sigma_MCMC(:, :, j) = Sigma0/(nu-d-1);
    det_Sigma_MCMC(j) = det(Sigma_MCMC(:,:,j));
    [L_Sigma,D_Sigma] = ldl(Sigma_MCMC(:,:,j));
    diag_D_MCMC(:,j) = diag(D_Sigma);
    LD_Sigma_Vec_MCMC(1,j) = log(D_Sigma(1,1));
    for i1 = 1:(d-1)
        LD_Sigma_Vec_MCMC((d*(i1-1)+i1-i1*(i1-1)/2+1):(d*i1-i1*(i1-1)/2+1),j) = [L_Sigma((i1+1):d,i1)' log(D_Sigma(i1+1,i1+1))];
    end
end
% Set initial values for kronecker product: Careful with Kronecker product!
%  Can rewrite in higher dimensions to avoid kronecker
Kron_Sj = zeros((p+q+1) * d, (p+q+1) * d, it);
for j = 1:it
    Kron_Sj(:,:,j) = kron(Sigma_MCMC(:, :, j),U_iC);
end

%Initial value of beta= prior mean across components
% rearrange matrix
beta0_Vec = reshape(beta0,(p+q+1)*d,1);
% generate sample
beta_MCMC = zeros((p+q+1), d, it);
beta_MCMC_Vec = zeros((p+q+1)*d,it);
for j = 1:it
    Kron_Sj(:,:,j) = kron(Sigma_MCMC(:, :, j),U_iC);
    beta_MCMC(:,:,j) = beta0;
    beta_MCMC_Vec(:,j) =reshape(beta_MCMC(:,:,j),(p+q+1)*d,1);
end

if p>0
    %Initial value of tau = prior sample
    tau_MCMC = gamrnd(repmat(a1',1,it),repmat(1./a2',1,it));

    %Initial value of mu= prior sample
    mu_MCMC = repmat(mu0',1,it)+randn(p,it).*(tau_MCMC.^(-.5)).*repmat(ic'.^(-.5),1,it);
end

%initial Vs for the DP are iid Beta (M is the mass parameter)
V_MCMC = betarnd(1, M, 1, it);
%Computing the weights for the DP
W_MCMC = zeros(1, it);
prodV = 1;
for j = 1:it
    W_MCMC(j) = V_MCMC(j) * prodV;
    prodV = prodV * (1 - V_MCMC(j));
end

if q>0
    %Initial value of rho = prior sample
    rho_MCMC = zeros(q,it);
    for h = 1:q
        aux=gamrnd(repmat(alpha_rho(:,h),1,it),1);
        rho_MCMC(h,:)=aux(1,:)./(aux(1,:)+aux(2,:));
    end
end

%Initial values of loglikelihood elements
num_MCMC = zeros(n,it);
unnorm_MCMC = zeros(n,it);
for j = 1:it
    if p>0
        unnorm_MCMC(:,j) = log(W_MCMC(j)) + log( mvnpdf(x(:,1:p),repmat(mu_MCMC(:,j)',n,1),diag(tau_MCMC(:,j).^(-1),0)) );
    end
    if q>0
        for h = 1:q
            unnorm_MCMC(:,j) = unnorm_MCMC(:,j) + (x(:,p+h) == 1)*log(rho_MCMC(h,j))+(x(:,p+h) == 2)*log(1-rho_MCMC(h,j));
        end
    end
    num_MCMC(:,j) = mvnpdf(y_MCMC,X*beta_MCMC(:,:,j),Sigma_MCMC(:,:,j));
end
norm_weights_MCMC = exp(unnorm_MCMC - repmat(max(unnorm_MCMC,[],2),1,it))./repmat(sum(exp(unnorm_MCMC - repmat(max(unnorm_MCMC,[],2),1,it)),2),1,it);
loglike_MCMC = sum(log(sum(num_MCMC.*norm_weights_MCMC,2)));


%% Adaptive Random Walk Initialisations

% Initialise log-variances of the adaptive MH for each block of
% parameters
tysd = zeros(n,d,d);
LDSigmasd = zeros(d*(d+1)/2,d*(d+1)/2,it);
betasd = zeros((p+q+1)*d,(p+q+1)*d,it);
if p>0
    tausd = zeros(p,p,it);
    musd = zeros(p,p,it);
end
Vsd = ones(1,it);
if q>0
    rhosd = ones(q,it);
end
for i = 1:n
    tysd(i,:,:) = 1.5*eye(d);
end
for j = 1:it
    LDSigmasd(:,:,j) = 0.1*eye(d*(d+1)/2);
    betasd(:,:,j) = 0.1*eye((p+q+1)*d);
    if p>0
        tausd(:,:,j) = eye(p);
        musd(:,:,j) = eye(p);
    end
end

%Intialise cummulative sum of transformed parameters for each block of
%parameters (used to iteratively update sd in adaptive MH)
sumty = zeros(n,d);
sumLDSigma = zeros(d*(d+1)/2,it);
sumbeta = zeros((p+q+1)*d,it);
if p>0
    sumtau = zeros(p,it);
    summu = zeros(p,it);
end
sumV = zeros(1,it);
if q>0
    sumrho = zeros(q,it);
end

%Intialise cummulative product of transformed parameters for each block of
%parameters (used to iteratively update sd in adaptive MH)
prodty = zeros(n,d,d);
prodLDSigma = zeros(d*(d+1)/2,d*(d+1)/2,it);
prodbeta = zeros((p+q+1)*d,(p+q+1)*d,it);
if p>0
    prodtau = zeros(p,p,it);
    prodmu = zeros(p,p,it);
end
prodV = zeros(1,it);
if q>0
    prodrho = zeros(q,it);
end

% Initialise new LDL decomposition in proposal of random walk
newL_Sigma = eye(d);
newD_Sigma = zeros(d,d);

%Initial new numerator and unnormalised calcusation of likelihood in
%proposal of random walk
newnum_MCMC = num_MCMC;
newunnorm_MCMC = unnorm_MCMC;

%acceptance rate and counters for NaN/Inf
tyaccept = zeros(1,n);
tycount = zeros(1,n);
Sigmaaccept = zeros(1,it);
Sigmacount = zeros(1,it);
betaaccept = zeros(1,it);
betacount = zeros(1,it);
if p>0
    tauaccept = zeros(1,it);
    taucount = zeros(1,it);
    muaccept = zeros(1,it);
    mucount = zeros(1,it);
end
Vaccept = zeros(1,it);
Vcount = zeros(1,it);
if q>0
    rhoaccept = zeros(q,it);
    rhocount = zeros(q,it);
end

%Adaptive proposals parameters
sd_ty = zeros(numbofits+1,n);
sd_LDSigma = zeros(numbofits+1,it);
sd_beta = zeros(numbofits+1,it);
if p>0
    sd_tau = zeros(numbofits+1,it);
    sd_mu = zeros(numbofits+1,it);
end
sd_V = zeros(numbofits+1,it);
if q>0
    sd_rho = zeros(q,numbofits+1,it);
end

sd_ty(1,:) = 2.4^2/d*ones(1,n);
sd_LDSigma(1,1:it) = 2.4^2/(d*(d+1)/2);
sd_beta(1,1:it) = 2.4^2/((p+q+1)*d);
if p>0
    sd_tau(1,1:it) = 2.4^2/p;
    sd_mu(1,1:it) = 2.4^2/p;
end
sd_V(1,1:it) = 2.4^2;
if q>0
    sd_rho(1:q,1,1:it) = 2.4^2;
end

%Vector of adapting paramters used in every iteration
%Apart from "accept" (ADAPT(3)) that changes
ADAPT = zeros(1,5);
ADAPT(5) = 0.001; % nugget to ensure inverbility and minimal level of exploration
ADAPT(4) = 0.234; % targeted acceptance level
ADAPT(2) = .7;
ADAPT(1) = 100; % start adapting after ADAPT(1) iterations

%% MCMC part to produce particles at initial truncation level

for it2 = 1:numbofits
    
    %% Update latent y
    for i = 1:n
        % Random walk proposal: transform y to real line
        tyi = ty(y_MCMC(i,:),lower(i,:),upper(i,:));
        ty_new = tyi + randn(1,d) * cholcov(squeeze(tysd(i,:,:)));
        y_new = Inv_ty(ty_new, lower(i,:), upper(i,:));
        
        % Compute acceptance ratio: likelihood part
        % Compute local multivariate regression for new y_i
        for j = 1:it
            newnum_MCMC(i,j) = mvnpdf(y_new,X(i,:)*beta_MCMC(:,:,j),Sigma_MCMC(:,:,j));
        end
        % Since y_i are conditionally independent, ratio of loglikelihood
        % only involves likelihood at y_i
        logaccept = log(sum(newnum_MCMC(i,:).*norm_weights_MCMC(i,:))) - log(sum(num_MCMC(i,:).*norm_weights_MCMC(i,:)));
        
        % Compute acceptance ratio: add prior and proposal part
        logaccept = logaccept + ty_ratio(y_MCMC(i,:),y_new,lower(i,:),upper(i,:));
        
        % Transform log acceptance ratio to probability
        if (isreal(logaccept) == 0 )
            stop(1);
        end        
        accept = 1;
        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
            accept = 0;
        elseif ( logaccept < 0 )
            accept = exp(logaccept);
        end
        
        % Average acceptance rate: update elements
        tyaccept(i) = tyaccept(i) + accept;
        tycount(i) = tycount(i) + 1;
        
        % Accept/Reject Move
        if ( rand < accept )
            %Accept and set new values of y and likelihood components
            y_MCMC(i,:) = y_new;
            num_MCMC(i,:) = newnum_MCMC(i,:);
        else
            %Reject proposed likelihood components are returned to old
            %values
            newnum_MCMC(i,:) = num_MCMC(i,:);
        end
        
        % Update parameters of adaptive random walk
        ADAPT(3) = accept;
        [sd_ty(it2+1,i),sumty(i,:),prodty(i,:,:),tysd(i,:,:)] = Alg6(tyi', sumty(i,:)', squeeze(prodty(i,:,:)), squeeze(tysd(i,:,:)), sd_ty(it2,i), it2, ADAPT);
    end
    
    % Recompute loglikehood
    loglike_MCMC = sum(log(sum(num_MCMC.*norm_weights_MCMC,2)));
        
    %% For every component: update parameters
    for j = 1:it
        
        %% Update Sigma
        % Random walk proposal: based log LDL transform
        newLD_Sigma_Vec = LD_Sigma_Vec_MCMC(:,j) + (randn(1,d*(d+1)/2) * cholcov(LDSigmasd(:,:,j)))';
        newD_Sigma(1,1) = exp(newLD_Sigma_Vec(1));
        for i1 = 1:(d-1)
            newL_Sigma((i1+1):d,i1) = newLD_Sigma_Vec((d*(i1-1)+i1-i1*(i1-1)/2+1):(d*i1-i1*(i1-1)/2));
            newD_Sigma(i1+1,i1+1) = exp(newLD_Sigma_Vec(d*i1-i1*(i1-1)/2+1));
        end
        newdiag_D = diag(newD_Sigma);
        newSigma_MCMC = newL_Sigma * newD_Sigma * newL_Sigma';
        newdet_Sigma = det(newSigma_MCMC);
        
        % Compute acceptance ratio: likelihood part
        % Only need to recompute the local multivariate linear regression
        newnum_MCMC(:,j) = mvnpdf(y_MCMC,X*beta_MCMC(:,:,j),newSigma_MCMC);
        newloglike_MCMC = sum(log(sum(newnum_MCMC.*norm_weights_MCMC,2)));
        logaccept = newloglike_MCMC - loglike_MCMC;
        
        % Compute acceptance ratio: prior and proposal part
        % Note: Kronecker product can be avoided in higher dimensions
        newKron_Sj = kron(newSigma_MCMC,U_iC);
        %Note: the prior for beta_j contains the det(Sigma_j) (MV normal), so include 2*d
        logaccept = logaccept + (d:-1:1) * (log(newdiag_D)-log(diag_D_MCMC(:,j))) - (nu+d+p+q+2)/2*(log(newdet_Sigma)-log(det_Sigma_MCMC(j))) -.5*( trace((inv(newSigma_MCMC) - inv(Sigma_MCMC(:,:,j)))*Sigma0) +  (beta_MCMC_Vec(:,j) - beta0_Vec)'/newKron_Sj*(beta_MCMC_Vec(:,j) - beta0_Vec) - (beta_MCMC_Vec(:,j) - beta0_Vec)'/Kron_Sj(:,:,j)*(beta_MCMC_Vec(:,j) - beta0_Vec) );   
        
        % Transform log acceptance ratio to probability
        if (isreal(logaccept) == 0 )
            stop(1);
        end
        accept = 1;
        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
            accept = 0;
        elseif ( logaccept < 0 )
            accept = exp(logaccept);
        end
        
        % Average acceptance rate: update elements
        Sigmaaccept(j) = Sigmaaccept(j) + accept;
        Sigmacount(j) = Sigmacount(j) + 1;
        
        % Accept/Reject Move
        if ( rand < accept )
            %Accept and set new values of Sigma and likelihood components
            Sigma_MCMC(:,:,j) = newSigma_MCMC;
            LD_Sigma_Vec_MCMC(:,j) = newLD_Sigma_Vec;
            diag_D_MCMC(:,j) = newdiag_D;
            det_Sigma_MCMC(j) = newdet_Sigma;
            loglike_MCMC = newloglike_MCMC;
            num_MCMC(:,j) = newnum_MCMC(:,j);
            Kron_Sj(:,:,j) = newKron_Sj;
        else
            %Reject proposed likelihood components are returned to old
            %values
            newnum_MCMC(:,j) = num_MCMC(:,j);
        end
        
        % Update parameters of adaptive random walk
        ADAPT(3) = accept;
        [sd_LDSigma(it2+1,j),sumLDSigma(:,j),prodLDSigma(:,:,j),LDSigmasd(:,:,j)] = Alg6(LD_Sigma_Vec_MCMC(:,j), sumLDSigma(:,j), prodLDSigma(:,:,j), LDSigmasd(:,:,j), sd_LDSigma(it2,j), it2, ADAPT);
          
        %% Update beta
        
        % % Random walk proposal: no transformation needed
        newbeta_MCMC_Vec = beta_MCMC_Vec(:,j) + (randn(1,(p+q+1)*d) * cholcov(betasd(:,:,j)))';
        newbeta_MCMC = reshape(newbeta_MCMC_Vec,(p+q+1),d);
        
        % Compute acceptance ratio: likelihood part
        % Only need to recompute the local multivariate linear regression
        newnum_MCMC(:,j) = mvnpdf(y_MCMC,X*newbeta_MCMC,Sigma_MCMC(:,:,j));
        newloglike_MCMC = sum(log(sum(newnum_MCMC.*norm_weights_MCMC,2)));
        logaccept = newloglike_MCMC - loglike_MCMC;
        
        % Compute acceptance ratio: prior part
        % Note: Kronecker product can be avoided in higher dimensions
        logaccept = logaccept - .5 * ( (newbeta_MCMC_Vec - beta0_Vec)'/Kron_Sj(:,:,j)*(newbeta_MCMC_Vec - beta0_Vec) - (beta_MCMC_Vec(:,j) - beta0_Vec)'/Kron_Sj(:,:,j)*(beta_MCMC_Vec(:,j) - beta0_Vec) );
        
        % Transform log acceptance ratio to probability
        if (isreal(logaccept) == 0 )
            stop(1);
        end
        accept = 1;
        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
            accept = 0;
        elseif ( logaccept < 0 )
            accept = exp(logaccept);
        end
        
        % Average acceptance rate: update elements
        betaaccept(j) = betaaccept(j) + accept;
        betacount(j) = betacount(j) + 1;
        
        % Accept/Reject Move
        if ( rand < accept )
            %Accept and set new values of beta and likelihood components
            beta_MCMC(:,:,j) = newbeta_MCMC;
            beta_MCMC_Vec(:,j) = newbeta_MCMC_Vec;
            loglike_MCMC = newloglike_MCMC;
            num_MCMC(:,j) = newnum_MCMC(:,j);
        else
            %Reject proposed likelihood components are returned to old
            %values
            newnum_MCMC(:,j) = num_MCMC(:,j);
        end
        
        % Update parameters of adaptive random walk
        ADAPT(3) = accept;
        [sd_beta(it2+1,j),sumbeta(:,j),prodbeta(:,:,j),betasd(:,:,j)] = Alg6(beta_MCMC_Vec(:,j), sumbeta(:,j), prodbeta(:,:,j), betasd(:,:,j), sd_beta(it2,j), it2, ADAPT);
        
        if p>0
        %% Update tau
        
        
            % % Random walk proposal: based on log transformation
            newtau_MCMC = tau_MCMC(:,j).* exp(randn(1,p) * cholcov(tausd(:,:,j)))';
        
            % Compute acceptance ratio: likelihood part
            % Only need to recompute the unnormalised and normalised weights 
            newunnorm_MCMC(:,j) = unnorm_MCMC(:,j) - log( mvnpdf(x(:,1:p),repmat(mu_MCMC(:,j)',n,1),diag(tau_MCMC(:,j).^(-1),0)) ) + log( mvnpdf(x(:,1:p),repmat(mu_MCMC(:,j)',n,1),diag(newtau_MCMC.^(-1),0)) );
            newnorm_weights_MCMC = exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it))./repmat(sum(exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it)),2),1,it);
            newloglike_MCMC = sum(log(sum(num_MCMC.*newnorm_weights_MCMC,2)));
            logaccept = newloglike_MCMC - loglike_MCMC;
        
            % Compute acceptance ratio: prior and proposal part
            logaccept = logaccept + (a1 + .5)*(log(newtau_MCMC) - log(tau_MCMC(:,j))) - ((.5 .* ic).*(mu_MCMC(:,j)' - mu0).^2 + a2) * (newtau_MCMC - tau_MCMC(:,j));
        
            % Transform log acceptance ratio to probability
            if (isreal(logaccept) == 0 )
                stop(1);
            end        
            accept = 1;
            if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                accept = 0;
            elseif ( logaccept < 0 )
                accept = exp(logaccept);
            end
        
            % Average acceptance rate: update elements
            tauaccept(j) = tauaccept(j) + accept;
            taucount(j) = taucount(j) + 1;
        
            % Accept/Reject Move
            if ( rand < accept )
                %Accept and set new values of tau and (un)normalised weights
                tau_MCMC(:,j) = newtau_MCMC;
                loglike_MCMC = newloglike_MCMC;
                unnorm_MCMC(:,j) = newunnorm_MCMC(:,j);
                norm_weights_MCMC = newnorm_weights_MCMC;
            else
                %Reject proposed (un)normalised weights are returned to old
                %values
                newunnorm_MCMC(:,j) = unnorm_MCMC(:,j);
            end
        
            % Update parameters of adaptive random walk
            ADAPT(3) = accept;
            [sd_tau(it2+1,j),sumtau(:,j),prodtau(:,:,j),tausd(:,:,j)] = Alg6(log(tau_MCMC(:,j)), sumtau(:,j), prodtau(:,:,j), tausd(:,:,j), sd_tau(it2,j), it2, ADAPT);
                
            %% Update mu
        
            % % Random walk proposal: no transformation needed
            newmu_MCMC = mu_MCMC(:,j) + (randn(1,p) * cholcov(musd(:,:,j)))';
        
            % Compute acceptance ratio: likelihood part
            % Only need to recompute the unnormalised and normalised weights 
            newunnorm_MCMC(:,j) = unnorm_MCMC(:,j) - log( mvnpdf(x(:,1:p),repmat(mu_MCMC(:,j)',n,1),diag(tau_MCMC(:,j).^(-1),0)) ) + log( mvnpdf(x(:,1:p),repmat(newmu_MCMC',n,1),diag(tau_MCMC(:,j).^(-1),0)) );
            newnorm_weights_MCMC = exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it))./repmat(sum(exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it)),2),1,it);
            newloglike_MCMC = sum(log(sum(num_MCMC.*newnorm_weights_MCMC,2)));
            logaccept = newloglike_MCMC - loglike_MCMC;
               
            % Compute acceptance ratio: prior and proposal part
            logaccept = logaccept - .5 * (tau_MCMC(:,j)' .* ic) * ((newmu_MCMC - mu0').^2 - (mu_MCMC(:,j) - mu0').^2);
        
            % Transform log acceptance ratio to probability
            if (isreal(logaccept) == 0 )
                stop(1);
            end        
            accept = 1;
            if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                accept = 0;
            elseif ( logaccept < 0 )
                accept = exp(logaccept);
            end
        
            % Average acceptance rate: update elements
            muaccept(j) = muaccept(j) + accept;
            mucount(j) = mucount(j) + 1;
        
            % Accept/Reject Move
            if ( rand < accept )
                %Accept and set new values of mu and (un)normalised weights
                mu_MCMC(:,j) = newmu_MCMC;
                loglike_MCMC = newloglike_MCMC;
                unnorm_MCMC(:,j) = newunnorm_MCMC(:,j);
                norm_weights_MCMC = newnorm_weights_MCMC;
            else
                %Reject proposed (un)normalised weights are returned to old
                %values
                newunnorm_MCMC(:,j) = unnorm_MCMC(:,j);
            end
        
            % Update parameters of adaptive random walk
            ADAPT(3) = accept;
            [sd_mu(it2+1,j),summu(:,j),prodmu(:,:,j),musd(:,:,j)] = Alg6(mu_MCMC(:,j), summu(:,j), prodmu(:,:,j), musd(:,:,j), sd_mu(it2,j), it2, ADAPT);
        end 
        
        %% Update v
        
        % % Random walk proposal: based on logit transformation
        newV_MCMC = V_MCMC;
        trans = log(V_MCMC(j)) - log(1 - V_MCMC(j));
        newtrans = trans + randn * sqrt(Vsd(j));
        %Transform back
        newV_MCMC(j) = 1 / (1 + exp(-newtrans));        
        %Compute new set of weights (note: weights change for all
        %components with indices >=j)
        newW_MCMC = W_MCMC;
        if j>1
            for i = j:it
                newW_MCMC(i) = newV_MCMC(i) * (1 - newV_MCMC(i-1)) / newV_MCMC(i-1) * newW_MCMC(i-1);
            end
        else
            prodV_aux = 1;
            for i = 1:it
                newW_MCMC(i) = newV_MCMC(i) * prodV_aux;
                prodV_aux = prodV_aux * (1 - newV_MCMC(i));
            end
        end
        
        % Compute acceptance ratio: likelihood part
        % Only need to recompute the unnormalised and normalised weights 
        for i = j:it
            newunnorm_MCMC(:,i) = unnorm_MCMC(:,i) - log(W_MCMC(i)) + log(newW_MCMC(i));
        end
        newnorm_weights_MCMC = exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it))./repmat(sum(exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it)),2),1,it);
        newloglike_MCMC = sum(log(sum(num_MCMC.*newnorm_weights_MCMC,2)));
        logaccept = newloglike_MCMC - loglike_MCMC;
        
        % Compute acceptance ratio: prior and proposal part
        logaccept = logaccept + log(newV_MCMC(j))-log(V_MCMC(j)) + M * (log(1 - newV_MCMC(j)) - log(1 - V_MCMC(j)));
        
        % Transform log acceptance ratio to probability
        if (isreal(logaccept) == 0 )
            stop(1);
        end        
        accept = 1;
        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
            accept = 0;
        elseif ( logaccept < 0 )
            accept = exp(logaccept);
        end
        
        % Average acceptance rate: update elements
        Vaccept(j) = Vaccept(j) + accept;
        Vcount(j) = Vcount(j) + 1;
        
        % Accept/Reject Move
        if ( rand < accept )
            %Accept and set new values of v,w and (un)normalised weights
            V_MCMC(j) = newV_MCMC(j);
            W_MCMC = newW_MCMC;
            loglike_MCMC = newloglike_MCMC;
            for i = j:it
                unnorm_MCMC(:,i) = newunnorm_MCMC(:,i);
            end
            norm_weights_MCMC = newnorm_weights_MCMC;
        else
            %Reject proposed (un)normalised weights are returned to old
            %values
            for i = j:it
                newunnorm_MCMC(:,i) = unnorm_MCMC(:,i);
            end
        end
        
        % Update parameters of adaptive random walk
        ADAPT(3) = accept;
        [sd_V(it2+1,j), sumV(j), prodV(j), Vsd(j)] = Alg6(log(V_MCMC(j)) - log(1 - V_MCMC(j)), sumV(j), prodV(j), Vsd(j), sd_V(it2,j), it2, ADAPT);
        
        if q>0
            %% Update rho
            for h = 1:q
                % % Random walk proposal: based on logit transformation
                %Transform rho
                trans = log(rho_MCMC(h,j))-log(1-rho_MCMC(h,j));
                newtrans = trans + randn(1,1) * cholcov(rhosd(h,j));
                %Transform back
                newrho_MCMC = exp(newtrans) / (1 + exp(newtrans));
                
                % Compute acceptance ratio: likelihood part
                % Only need to recompute the unnormalised and normalised weights 
                newunnorm_MCMC(:,j) = newunnorm_MCMC(:,j) + (x(:,p+h) == 1)*( - log(rho_MCMC(h,j)) + log(newrho_MCMC) )+(x(:,p+h) == 2)*( - log(1-rho_MCMC(h,j)) + log(1-newrho_MCMC) );
                newnorm_weights_MCMC = exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it))./repmat(sum(exp(newunnorm_MCMC - repmat(max(newunnorm_MCMC,[],2),1,it)),2),1,it);
                newloglike_MCMC = sum(log(sum(num_MCMC.*newnorm_weights_MCMC,2)));
                logaccept = newloglike_MCMC - loglike_MCMC;
                
                % Compute acceptance ratio: prior and proposal part
                logaccept = logaccept + sum( alpha_rho(:,h).*([log(newrho_MCMC);log(1-newrho_MCMC)] - [log(rho_MCMC(h,j));log(1-rho_MCMC(h,j))]) );
                
                % Transform log acceptance ratio to probability
                if (isreal(logaccept) == 0 )
                    stop(1);
                end        
                accept = 1;
                if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                accept = 0;
                elseif ( logaccept < 0 )
                    accept = exp(logaccept);
                end
        
                % Average acceptance rate: update elements  
                rhoaccept(h,j) = rhoaccept(h,j) + accept;
                rhocount(h,j) = rhocount(h,j) + 1;
            
                % Accept/Reject Move
                if ( rand < accept )
                    %Accept and set new values of rho and (un)normalised weights
                    rho_MCMC(h,j) = newrho_MCMC;
                    loglike_MCMC = newloglike_MCMC;
                    unnorm_MCMC(:,j) = newunnorm_MCMC(:,j);
                    norm_weights_MCMC = newnorm_weights_MCMC;
                else
                    %Reject proposed (un)normalised weights are returned to old
                    %values            
                    newunnorm_MCMC(:,j) = unnorm_MCMC(:,j);
                end
            
                % Update parameters of adaptive random walk
                ADAPT(3) = accept;
                [sd_rho(h,it2+1,j), sumrho(h,j), prodrho(h,j), rhosd(h,j)] = Alg6(log(rho_MCMC(h,j)) - log(1-rho_MCMC(h,j)), sumrho(h,j), prodrho(h,j), rhosd(h,j), sd_rho(h,it2,j), it2, ADAPT);
            end
        end
    end
    
    %Displaying the values every 100 iterations
    if ( mod(it2, 100) == 0 )
        disp(['it2 = ' num2str(it2)]);
        disp(['Sigma accept: avg = ' num2str(mean(Sigmaaccept./Sigmacount)),', min = ',num2str(min(Sigmaaccept./Sigmacount)), ', max = ',num2str(max(Sigmaaccept./Sigmacount))]);
        disp(['beta accept: avg = ' num2str(mean(betaaccept./betacount)), ', min = ',num2str(min(betaaccept./betacount)), ', max = ',num2str(max(betaaccept./betacount))]);
        if p>0
            disp(['tau accept: avg = ' num2str(mean(tauaccept./taucount)), ', min = ',num2str(min(tauaccept./taucount)),', max = ',num2str(max(tauaccept./taucount))]);
            disp(['mu accept: avg = ' num2str(mean(muaccept./mucount)), ', min = ',num2str(min(muaccept./mucount)), ', max = ',num2str(max(muaccept./mucount))]);
        end
        disp(['V accept: avg = ' num2str(mean(Vaccept(1:end-1)./Vcount(1:end-1))), ', min = ',num2str(min(Vaccept(1:end-1)./Vcount(1:end-1))), ', max = ',num2str(max(Vaccept(1:end-1)./Vcount(1:end-1)))]);
        if q>0
            for h = 1:q
                disp(['rho_',num2str(h),' accept: avg = ' num2str(mean(rhoaccept(h,:)./rhocount(h,:))), ', min = ', num2str(min(rhoaccept(h,:)./rhocount(h,:))),', max = ', num2str(max(rhoaccept(h,:)./rhocount(h,:)))]);
            end
        end
        disp(['y accept: avg = ' num2str(mean(tyaccept./tycount)), ', min = ',num2str(min(tyaccept./tycount)), ', max = ',num2str(max(tyaccept./tycount))]);
        disp(' ');
    end
    
    
    %Saving the output after burn-in period and thinning "every" iteration    
    loglike_MCMC_save(it2) = loglike_MCMC;    
    if ( (it2 > burnin) && ( mod(it2 - burnin, every) == 0) )        
        iter_aux = (it2 - burnin) / every;
        y(iter_aux,:,:) = y_MCMC;
        for i1 = 1:(d-1)
            Sigma(iter_aux, 1:it,i1,i1) = Sigma_MCMC(i1,i1,:);
            for i2 = (i1+1):d
                Sigma(iter_aux, 1:it,i1,i2) = Sigma_MCMC(i1,i2,:);
                Sigma(iter_aux, 1:it,i2,i1) = Sigma_MCMC(i2,i1,:);
            end
        end
        Sigma(iter_aux, 1:it,d,d) = Sigma_MCMC(d,d,1:it);
        det_Sigma(iter_aux, 1:it) = det_Sigma_MCMC;
        for j = 1:it
            LD_Sigma_Vec(iter_aux, j, :) = LD_Sigma_Vec_MCMC(:,j);
            diag_D(iter_aux, j, :) = diag_D_MCMC(:,j);
        end
        for i1 = 1:(p+q+1)
            for i2 = 1:d
                beta_Vec(iter_aux, 1:it, (i1-1)*d + i2) = beta_MCMC_Vec((i1-1)*d + i2, :);
                beta(iter_aux, 1:it, i1, i2) = beta_MCMC(i1, i2, :);
            end
        end
        if p>0
            for i1 = 1:p
                tau(iter_aux, 1:it, i1) = tau_MCMC(i1,:);
                mu(iter_aux, 1:it, i1) = mu_MCMC(i1,:);
            end 
        end
        V(iter_aux, 1:it) = V_MCMC;
        W(iter_aux, 1:it) = W_MCMC;
        if q>0
            for h = 1:q
                for j = 1:it
                    rho(h,iter_aux,j) = rho_MCMC(h,j);
                end
            end
        end
        num(iter_aux,:,1:it) = num_MCMC;
        unnorm(iter_aux,:,1:it) = unnorm_MCMC;
        norm_weights(iter_aux,:,1:it) = norm_weights_MCMC;
        loglike(iter_aux) = loglike_MCMC;
    end
    
end

%% Sequential Monte Carlo part

loglike_SMC(:,1:it) = repmat(loglike,1,it);

% For rejuventation step: compute variance of random walk, as the average 
% of the adaptive variances over components, weighted using the components 
% weights
mean_W = mean(W(:,1:it)); %mean of component weights over the particles
mean_W = mean_W/sum(mean_W); %normalize
%compute average
aux_LDSigmasd = zeros(d*(d+1)/2,d*(d+1)/2);
aux_betasd = zeros((p+q+1)*d,(p+q+1)*d);
if p>0
    aux_tausd = zeros(p,p);
    aux_musd = zeros(p,p);
end
if q>0
    aux_rhosd = zeros(1,q);
end
for j = 1:it
    aux_LDSigmasd = aux_LDSigmasd + LDSigmasd(:,:,j)*mean_W(j);
    aux_betasd = aux_betasd + betasd(:,:,j)*mean_W(j);
    if p>0
        aux_tausd = aux_tausd + tausd(:,:,j)*mean_W(j);
        aux_musd = aux_musd + musd(:,:,j)*mean_W(j);
    end
    if q>0
        aux_rhosd = aux_rhosd+ rhosd(:,j)'* mean_W(j);
    end
end
LDSigmasd = aux_LDSigmasd;
betasd = aux_betasd;
if p>0
    tausd = aux_tausd;
    musd = aux_musd;
end
if q>0
    rhosd = aux_rhosd;
end
Vsd = sum(Vsd.* mean_W);
% Note: ysd is the same, as it doesn't depend on components

% auxilary particle indices of resampling step
partstar = zeros(1, numbofparts);
%rat contains the mean of the incremental weights
rat = zeros(1, top_trunc - start_trunc + 1);
% store the ESS across SMC iterations
storeESS = zeros(1, top_trunc - start_trunc + 1);

%log of the initial weights (they are all equal to 1)
logweight = zeros(numbofparts, top_trunc);

%when check_trunc==1, we met the stopping rule
check_trunc = 0;

%ESS_TYPE (=1 for ESS or =0 for CESS)
ess_type = 1;

%truncation level from which we start applying the SMC algorithm
it = start_trunc - 1;

while ( check_trunc == 0 && it < top_trunc )    
    %% Step 1: add new component, sample new component parameters from prior, update log likelihood and particle weights
    it = it + 1;
    
    % new loglikelihood for it=it+1 components
    newloglike = zeros(1, numbofparts);
    
    %Proposed values for new component from the prior (can be done in
    %parallel)
    for part = 1:numbofparts
        Sigma(part,it,:, :) = iwishrnd(Sigma0,nu);        
        det_Sigma(part,it) = det(squeeze(Sigma(part,it,:, :)));        
        [L_Sigma,D_Sigma] = ldl(squeeze(Sigma(part,it,:, :)));
        diag_D(part,it,:) = diag(D_Sigma);
        LD_Sigma_Vec(part,it,1) = log(D_Sigma(1,1));
        for i1 = 1:(d-1)
            LD_Sigma_Vec(part,it,(d*(i1-1)+i1-i1*(i1-1)/2+1):(d*i1-i1*(i1-1)/2+1)) = [L_Sigma((i1+1):d,i1)' log(D_Sigma(i1+1,i1+1))];
        end
        LD_Sigma_Vec(part,it,end) = log(D_Sigma(d,d));        
        %Note: can avoid the Kronecker product here
        Kron_Sj = kron(squeeze(Sigma(part,it,:, :)),U_iC);
        beta_Vec(part, it, :) = beta0_Vec + (randn(1,(p+q+1)*d) * cholcov(Kron_Sj))';
        beta(part, it, :, :) = reshape(beta_Vec(part, it, :),(p+q+1),d);
    end
    if p>0
        tau(:, it, :) = gamrnd(repmat(a1,numbofparts,1),repmat(1./a2,numbofparts,1));
        mu(:, it, :) = repmat(mu0,numbofparts,1) + (squeeze(tau(:, it, :)).^(-.5)).*repmat(ic.^(-.5),numbofparts,1).* randn(numbofparts, p);
    end
    %Finite mixture originated from SB process
    V(:, it) = betarnd(1, M, numbofparts, 1);
    %Update the weight including the new particle
    W(:,it) = V(:,it) .* (1-V(:,it-1)) ./ V(:,it-1) .* W(:,it-1);
    if q>0
        for h = 1:q
            for h1 = 1:numbofparts
                aux = gamrnd(alpha_rho(:,h),1);
                rho(h, h1, it) = aux(1)/sum(aux);
            end
        end
    end
    
    %Compute log-likelihood for each particle and update particle weights
    %(this can be done in parallel)
    for part = 1:numbofparts
        
        % Compute unnormalised weight of the new component for each particle
        if p>0
            unnorm(part,:,it) = log(W(part,it)) + log( mvnpdf(x(:,1:p),repmat(squeeze(mu(part,it,:))',n,1),diag(squeeze(tau(part,it,:)).^(-1),0)) );
        end
        if q>0
            for h = 1:q
                unnorm(part,:,it) = unnorm(part,:,it) + (x(:,p+h) == 1)'*log(rho(h,part,it))+ (x(:,p+h) == 2)'*log(1-rho(h,part,it));
            end
        end
        % Compute the local likelihood of the new component for each particle
        if d==1
            num(part,:,it) = mvnpdf(squeeze(y(part,:,:))',X*squeeze(beta(part,it,:,:)),squeeze(Sigma(part,it,:,:)));
        else
            num(part,:,it) = mvnpdf(squeeze(y(part,:,:)),X*squeeze(beta(part,it,:,:)),squeeze(Sigma(part,it,:,:)));
        end
        % Compute the normalised weights for each particle
        norm_weights(part,:,1:it) = exp(squeeze(unnorm(part,:,1:it)) - repmat(max(squeeze(unnorm(part,:,1:it)),[],2),1,it))./repmat(sum(exp(squeeze(unnorm(part,:,1:it)) - repmat(max(squeeze(unnorm(part,:,1:it)),[],2),1,it)),2),1,it);
        % Compute the new loglikelihood for each particle
        newloglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*squeeze(norm_weights(part,:,1:it)),2)));
        %Compute incremental term in the SMC scheme (alpha)
        logweight(part,it) = logweight(part,it-1) + newloglike(part) - loglike(part);
    end
    loglike = newloglike;
    
    %% Step 2: Compute Effective Sample Size/Conditional Effective Sample Size
    if ess_type == 1
        ESS = (sum(exp(logweight(:,it) - max(logweight(:,it)))))^2 / (sum(exp(2*(logweight(:,it) - max(logweight(:,it))))));
        ESS_SMC(it - start_trunc + 1,1) = ESS;
    else
        norm_w_prev = exp(logweight(:,it-1) - max(logweight(:,it-1)))./sum(exp(logweight(:,it-1) - max(logweight(:,it-1))));
        ESS = numbofparts * (sum(norm_w_prev .* exp(logweight(:,it) - max(logweight(:,it)))))^2 / (sum(norm_w_prev .* exp(2*(logweight(:,it) - max(logweight(:,it))))));
        ESS_SMC(it - start_trunc + 1,2) = ESS;
    end
    
    %Display the truncation level and the ESS
    disp(num2str(it));
    disp(['ESS = ' num2str(ESS)]);
    disp(' ');
    
    %rat contains the mean of the incremental weights
    rat(it - start_trunc + 1) = mean(exp(logweight(:,it)));
    
    %Saving the ESS for this truncation level
    storeESS(it - start_trunc + 1) = ESS;
    
    %% Step 3: Resampling/Rejuvenation: if ESS is too small
    if ( ESS < 0.5 * numbofparts )
        
        %Compute probabilites proportional to the current set of weights
        prob2 = exp(logweight(:,it) - max(logweight(:,it)));
        fprob = cumsum(prob2);
        fprob = fprob / fprob(end);
        
        %Systematic re-sampling
        % sample particle indices
        u1 = rand / numbofparts;
        j = 1;
        count =  1;
        while ( j <= numbofparts )
            while ( (u1 <= fprob(j)) && (j <= numbofparts) )
                partstar(count) = j;
                u1 = u1 + 1 / numbofparts;
                count = count + 1;
            end
            j = j + 1;
        end
        
        %set values according to sampled particle indices
        y = y(partstar,:,:);
        Sigma(:, 1:it, :, :) = Sigma(partstar, 1:it, :, :);
        det_Sigma(:, 1:it) = det_Sigma(partstar, 1:it);
        LD_Sigma_Vec(:, 1:it, :) = LD_Sigma_Vec(partstar, 1:it, :);
        diag_D(:, 1:it, :) = diag_D(partstar, 1:it, :);
        beta_Vec(:, 1:it, :) = beta_Vec(partstar, 1:it, :);
        beta(:, 1:it, :, :) = beta(partstar, 1:it, :, :);
        if p>0
            tau(:, 1:it, :) = tau(partstar, 1:it, :);
            mu(:, 1:it, :) = mu(partstar, 1:it, :);
        end
        V(:, 1:it) = V(partstar, 1:it);
        W(:, 1:it) = W(partstar, 1:it);
        if q>0
            rho(:,:,1:it) = rho(:,partstar,1:it);
        end
        loglike = newloglike(partstar);
        
        %If resampling, all the weights are the same (hence later
        %normalisation will not affect in case of CESS)
        logweight(:,it) = zeros(1, numbofparts);
        
        num(:,:,1:it) = num(partstar,:,1:it);
        unnorm(:,:,1:it) = unnorm(partstar,:,1:it);
        norm_weights(:,:,1:it) = norm_weights(partstar,:,1:it);
        
        % Rejuvenation Step
        % For MCMC proposal
        newnum = num;
        newunnorm = unnorm;
        
        %Re-initialise to zero acceptance rate variables
        %acceptance rate and counters for NaN/Inf
        tyaccept = zeros(1,n);
        tycount = zeros(1,n);
        Sigmaaccept = 0;
        Sigmacount = 0;
        betaaccept = 0;
        betacount = 0;
        if p>0
            tauaccept = 0;
            taucount = 0;
            muaccept = 0;
            mucount = 0;
        end
        Vaccept = 0;
        Vcount = 0;
        if q>0
            rhoaccept = zeros(1,q);
            rhocount = zeros(1,q);
        end
        
        %MCMC step
        %Perform numbofMCMC for each particle
        %Random walk with average variance from full MCMC
        
        % This can be done in parallel across particles
        for it2 = 1:numbofMCMC
            for part = 1:numbofparts
                
                %Update latent y's
                for i = 1:n
                    yi=y(part,i,:);
                    tyi=ty(yi,lower(i,:), upper(i,:));
                    ty_new = tyi + randn(1,d) * cholcov(squeeze(tysd(i,:,:)));
                    y_new = Inv_ty(ty_new, lower(i,:), upper(i,:));
                    
                    % Compute new log-likelihood
                    for j = 1:it
                        newnum(part,i,j) = mvnpdf(y_new,X(i,:)*squeeze(beta(part,j,:,:)),squeeze(Sigma(part,j,:,:)));
                    end
                    
                    % Compute acceptance ratio
                    % Loglikehood part: y_i are conditionally independent
                    logaccept = log(sum(squeeze(newnum(part,i,1:it)).*squeeze(norm_weights(part,i,1:it)))) - log(sum(squeeze(num(part,i,1:it)).*squeeze(norm_weights(part,i,1:it))));                    
                    %Prior and proposal part
                    prop_ratio = ty_ratio(squeeze(y(part,i,:))',y_new,lower(i,:),upper(i,:));
                    logaccept = logaccept + prop_ratio;                    
                    if (isreal(logaccept) == 0 )
                        stop(1);
                    end                    
                    accept = 1;
                    if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                        accept = 0;
                    elseif ( logaccept < 0 )
                        accept = exp(logaccept);
                    end                    
                    tyaccept(i) = tyaccept(i) + accept;
                    tycount(i) = tycount(i) + 1;
                    
                    % Accept/Reject proposal
                    if ( rand < accept )
                        y(part,i,:) = y_new;
                        num(part,i,:) = newnum(part,i,:);
                    else
                        newnum(part,i,:) = num(part,i,:);
                    end
                    
                end
                %Updated loglikelihood can be computed only once due to conditional indepepndence
                loglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*squeeze(norm_weights(part,:,1:it)),2)));
                
                for j = 1:it
                    
                    %Update Sigma%
                    %Propose new value
                    newLD_Sigma_Vec = squeeze(LD_Sigma_Vec(part,j,:));
                    newLD_Sigma_Vec = newLD_Sigma_Vec + (randn(1,d*(d+1)/2) * cholcov(LDSigmasd))';                    
                    newD_Sigma(1,1) = exp(newLD_Sigma_Vec(1));
                    for i1 = 1:(d-1)
                        newL_Sigma((i1+1):d,i1) = newLD_Sigma_Vec((d*(i1-1)+i1-i1*(i1-1)/2+1):(d*i1-i1*(i1-1)/2));
                        newD_Sigma(i1+1,i1+1) = exp(newLD_Sigma_Vec(d*i1-i1*(i1-1)/2+1));
                    end
                    newdiag_D = diag(newD_Sigma);
                    newSigma = newL_Sigma * newD_Sigma * newL_Sigma';
                    newdet_Sigma = det(newSigma);
                    
                    %Compute new loglikelihood (only need to up local
                    %likelihood)
                    newnum(part,:,j) = mvnpdf(squeeze(y(part,:,:)),X*squeeze(beta(part,j,:,:)),newSigma);
                    newloglike(part) = sum(log(sum(squeeze(newnum(part,:,1:it)).*squeeze(norm_weights(part,:,1:it)),2)));
                    
                    %Compute acceptance ratio
                    %Loglikelihood part
                    logaccept = newloglike(part) - loglike(part);                    
                    %Prior and proposal part
                    Kron_Sj = kron(squeeze(Sigma(part,j,:,:)),U_iC);
                    newKron_Sj = kron(squeeze(newSigma),U_iC);
                    logaccept = logaccept + (d:-1:1) * (log(newdiag_D)-log(squeeze(diag_D(part,j,:)))) - (nu+d+p+q+2)/2*(log(newdet_Sigma)-log(det_Sigma(part,j))) -.5 * ( trace((inv(newSigma)-inv(squeeze(Sigma(part,j,:,:))))*Sigma0) + (squeeze(beta_Vec(part,j,:)) - beta0_Vec)'/newKron_Sj*(squeeze(beta_Vec(part,j,:)) - beta0_Vec) - (squeeze(beta_Vec(part,j,:)) - beta0_Vec)'/Kron_Sj*(squeeze(beta_Vec(part,j,:)) - beta0_Vec) );                                        
                    accept = 1;
                    if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                        accept = 0;
                    elseif ( logaccept < 0 )
                        accept = exp(logaccept);
                    end                   
                    Sigmaaccept = Sigmaaccept + accept;
                    Sigmacount = Sigmacount + 1;
                    
                    % Accept/reject proposal
                    if ( rand < accept )
                        Sigma(part,j,:,:) = newSigma;
                        LD_Sigma_Vec(part,j,:) = newLD_Sigma_Vec;
                        diag_D(part,j,:) = newdiag_D;
                        det_Sigma(part,j) = newdet_Sigma;
                        loglike(part) = newloglike(part);
                        num(part,:,j) = newnum(part,:,j);
                    else
                        newnum(part,:,j) = num(part,:,j);
                    end
                    
                    %Update beta%
                    %Propose new beta from multivariate normal proposal
                    newbeta_Vec = squeeze(beta_Vec(part,j,:)) + (randn(1,(p+q+1)*d) * cholcov(betasd))';
                    newbeta = reshape(newbeta_Vec,(p+q+1),d);
                    
                    %Compute new loglikelihood
                    newnum(part,:,j) = mvnpdf(squeeze(y(part,:,:)),X*newbeta,squeeze(Sigma(part,j,:,:)));
                    newloglike(part) = sum(log(sum(squeeze(newnum(part,:,1:it)).*squeeze(norm_weights(part,:,1:it)),2)));
                    
                    %Compute acceptance ratio
                    %Loglikelihood part
                    logaccept = newloglike(part) - loglike(part);                   
                    %Prior and proposal part
                    Kron_Sj = kron(squeeze(Sigma(part,j,:,:)),U_iC);
                    logaccept = logaccept - .5 * ((newbeta_Vec - beta0_Vec)'/Kron_Sj*(newbeta_Vec - beta0_Vec)-(squeeze(beta_Vec(part,j,:)) - beta0_Vec)'/Kron_Sj*(squeeze(beta_Vec(part,j,:)) - beta0_Vec));                    
                    accept = 1;
                    if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                        accept = 0;
                    elseif ( logaccept < 0 )
                        accept = exp(logaccept);
                    end                    
                    betaaccept = betaaccept + accept;
                    betacount = betacount + 1;
                    
                    % Accept/reject proposal
                    if ( rand < accept )
                        beta_Vec(part,j,:) = newbeta_Vec;
                        beta(part,j,:,:) = newbeta;
                        loglike(part) = newloglike(part);
                        num(part,:,j) = newnum(part,:,j);
                    else
                        newnum(part,:,j) = num(part,:,j);
                    end
                    
                    if p>0
                        %Update tau%
                        %Propose new value from lognormal proposal
                        if p==1
                            tau_aux = tau(part,j,:);
                            mu_aux = mu(part,j,:);
                        else
                            tau_aux = squeeze(tau(part,j,:))';
                            mu_aux = squeeze(mu(part,j,:))';
                        end
                        newtau = tau_aux.* exp(randn(1,p) * cholcov(tausd));
                    
                        %Compute new loglikelihood
                        newunnorm(part,:,j) = unnorm(part,:,j) + ( - log(mvnpdf(x(:,1:p),repmat(mu_aux,n,1),diag((tau_aux).^(-1),0))) + log(mvnpdf(x(:,1:p),repmat(mu_aux,n,1),diag(newtau.^(-1),0))) )';
                        newnorm_weights = exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it))./repmat(sum(exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it)),2),1,it);
                        newloglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*newnorm_weights,2)));
                    
                        %Compute acceptance ratio
                        %Loglikelihood part
                        logaccept = newloglike(part) - loglike(part);                    
                        %Prior and proposal part
                        logaccept = logaccept + (a1 + .5) * (log(newtau) - log(tau_aux))' - (.5 .* ic .* (mu_aux - mu0).^2 + a2) * (newtau-tau_aux)';
                        accept = 1;
                        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                            accept = 0;
                        elseif ( logaccept < 0 )
                            accept = exp(logaccept);
                        end                    
                        tauaccept = tauaccept + accept;
                        taucount = taucount + 1;
                    
                        % Accept/Reject proposal
                        if ( rand < accept )
                            tau(part,j,:) = newtau;
                            loglike(part) = newloglike(part);
                            unnorm(part,:,j) = newunnorm(part,:,j);
                            norm_weights(part,:,1:it) = newnorm_weights;
                        else
                            newunnorm(part,:,j) = unnorm(part,:,j);
                        end
                                       
                        %Update mu%
                        %Propose new mu from multivariate normal proposal
                        if p==1
                            tau_aux = tau(part,j,1);
                            mu_aux = mu(part,j,1);
                        else
                            tau_aux = squeeze(tau(part,j,:))';
                            mu_aux = squeeze(mu(part,j,:))';
                        end
                        newmu = mu_aux + randn(1,p) * cholcov(musd);
                    
                        %Compute new loglikelihood
                        newunnorm(part,:,j) = unnorm(part,:,j) + ( - log(mvnpdf(x(:,1:p),repmat(mu_aux,n,1),diag(tau_aux.^(-1),0))) + log(mvnpdf(x(:,1:p),repmat(newmu,n,1),diag(tau_aux.^(-1),0))) )';
                        newnorm_weights = exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it))./repmat(sum(exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it)),2),1,it);
                        newloglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*newnorm_weights,2)));
                    
                        %Compute acceptance ratio
                        %Loglikelihood part
                        logaccept = newloglike(part) - loglike(part);
                        %Prior and proposal part
                        logaccept = logaccept - .5 * (tau_aux .* ic) * ((newmu - mu0).^2 - (mu_aux - mu0).^2)' ;                    
                        accept = 1;
                        if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                            accept = 0;
                        elseif ( logaccept < 0 )
                            accept = exp(logaccept);
                        end                    
                        muaccept = muaccept + accept;
                        mucount = mucount + 1;
                    
                        %Accept/Reject proposal
                        if ( rand < accept )
                            mu(part, j, :) = newmu;
                            loglike(part) = newloglike(part);
                            unnorm(part,:,j) = newunnorm(part,:,j);
                            norm_weights(part,:,1:it) = newnorm_weights;
                        else
                            newunnorm(part,:,j) = unnorm(part,:,j);
                        end
                    end
                                       
                    %Update V%
                    %Propose new V from a logistic normal 
                    newV = V(part, 1:it);
                    trans = log(V(part, j)) - log(1 - V(part, j));
                    newtrans = trans + sqrt(Vsd) * randn;
                    newV(j) = 1 / (1 + exp(-newtrans));
                    
                    %Compute new W
                    newW = W(part, 1:it);
                    if j>1
                        for i = j:it
                            newW(i) = newV(i) * (1 - newV(i-1))/newV(i-1)*newW(i-1);
                        end
                    else
                        prodV_aux = 1;
                        for i = 1:it
                            newW(i) = newV(i) * prodV_aux;
                            prodV_aux = prodV_aux * (1 - newV(i));
                        end
                    end
                    
                    %Compute new loglikelihood
                    for i = j:it
                        newunnorm(part,:,i) = unnorm(part,:,i) - log(W(part,i)) + log(newW(i));
                    end
                    newnorm_weights = exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it))./repmat(sum(exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it)),2),1,it);
                    newloglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*newnorm_weights,2)));
                    
                    %Compute acceptance ratio
                    %Loglikelihood part
                    logaccept = newloglike(part) - loglike(part);                    
                    %Prior and proposal part
                    logaccept = logaccept + log(newV(j))-log(V(part,j)) + M * (log(1 - newV(j)) - log(1 - V(part,j)));                    
                    accept = 1;
                    if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                        accept = 0;
                    elseif ( logaccept < 0 )
                        accept = exp(logaccept);
                    end                    
                    Vaccept = Vaccept + accept;
                    Vcount = Vcount + 1;
                    
                    % Accept/Reject proposal
                    if ( rand < accept )
                        V(part, j) = newV(j);
                        W(part, 1:it) = newW;
                        loglike(part) = newloglike(part);
                        for i = j:it
                            unnorm(part,:,i) = newunnorm(part,:,i);
                        end
                        norm_weights(part,:,1:it) = newnorm_weights;
                    else
                        for i = j:it
                            newunnorm(part,:,i) = unnorm(part,:,i);
                        end
                    end
                    
                    %Update rho%
                    if q>0
                        for h = 1:q
                            %Propose new rho from a logistic normal 
                            trans = log(rho(h,part,j)) - log(1-rho(h,part,j));
                            newtrans = trans + randn(1,1) * cholcov(rhosd(h));
                            newrho= exp(newtrans) / (1 + exp(newtrans));
                            
                            %New log-likelihood
                            newunnorm(part,:,j) = newunnorm(part,:,j) + (x(:,p+h) == 1)'*( - log(rho(h,part,j)) + log(newrho))+ (x(:,p+h) == 2)'*( - log(1-rho(h,part,j)) + log(1-newrho));
                            newnorm_weights = exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it))./repmat(sum(exp(squeeze(newunnorm(part,:,1:it)) - repmat(max(squeeze(newunnorm(part,:,1:it)),[],2),1,it)),2),1,it);
                            newloglike(part) = sum(log(sum(squeeze(num(part,:,1:it)).*newnorm_weights,2)));
                            
                            %Compute acceptance ratio:
                            %log-likelihood part
                            logaccept = newloglike(part) - loglike(part);                            
                            %Prior part and proposal part
                            logaccept = logaccept + sum( alpha_rho(:,h).*([log(newrho_MCMC);log(1-newrho_MCMC)] - [log(rho(h,part,j)); log(1-rho(h,part,j))]) );
                            accept = 1;
                            if ( (isnan(logaccept) == 1) || (isinf(logaccept) == 1) )
                                accept = 0;
                            elseif ( logaccept < 0 )
                                accept = exp(logaccept);
                            end
                            rhoaccept(h) = rhoaccept(h) + accept;
                            rhocount(h) = rhocount(h) + 1;
                         
                            %Accept/Reject proposal
                            if ( rand < accept )
                                rho(h,part,j) = newrho;
                                loglike(part) = newloglike(part);
                                unnorm(part,:,j) = newunnorm(part,:,j);
                                norm_weights(part,:,1:it) = newnorm_weights;
                            else
                                newunnorm(part,:,j) = unnorm(part,:,j);
                            end
                        end 
                    end
                end
                                
                %Displaying the values every ten iterations
                if ( mod(part, 100) == 0 )
                    disp(['n_comp = ' num2str(it)]);
                    disp(['n_mcmc = ' num2str(it2)]);
                    disp(['part = ' num2str(part)]);
                    disp(['Sigma accept: ' num2str(Sigmaaccept./Sigmacount)]);
                    disp(['beta accept: ' num2str(betaaccept./betacount)]);
                    if p>0
                        disp(['tau accept: ' num2str(tauaccept./taucount)]);
                        disp(['mu accept: ' num2str(muaccept./mucount)]);
                    end
                    disp(['V accept: ' num2str(Vaccept./Vcount)]);
                    if q>0
                        for h = 1:q
                            disp(['rho_',num2str(h),' accept: ' num2str(rhoaccept(h)./rhocount(h))]);
                        end
                    end
                    disp(['y accept: avg = ' num2str(mean(tyaccept./tycount))]);
                    disp(' ');
                end                
            end
        end
                
        %Display acceptance rates
        disp(['Sigma accept = ' num2str(Sigmaaccept/Sigmacount)]);
        disp(['beta accept = ' num2str(betaaccept/betacount)]);
        if p>0
            disp(['tau accept = ' num2str(tauaccept/taucount)]);
            disp(['mu accept = ' num2str(muaccept/mucount)]);
        end
        disp(['V accept = ' num2str(Vaccept/Vcount)]);
        if q>0
            disp(['rho accept = ' num2str(rhoaccept./rhocount)]);
        end
        disp(['y accept = ' num2str(mean(tyaccept./tycount))]);
        disp(' ');        
    end
    
    
    %% Stopping rule
    % Compute the indexes for the two ESS that we need in the stopping rule (k and k-1)
    %then check if the stopping rule is met    
    if ( it - start_trunc >= numb_trunc )
        x1 = (it - numb_trunc + 1 - start_trunc + 1):(it - start_trunc + 1);
        x2 = (it - numb_trunc - start_trunc + 1):(it - 1 - start_trunc + 1);
        check_trunc = sum(abs(storeESS(x1) - storeESS(x2)) < epsilon_trunc * numbofparts) == numb_trunc;
    end
  loglike_SMC(:, it) = loglike;
  
end


%Save output
if q>0
    rho=rho(:,:,1:it);
end
Sigma = Sigma(:,1:it,:,:);
beta = beta(:,1:it,:,:);
if p>0
    tau = tau(:,1:it,:);
    mu = mu(:,1:it,:);
end
V = V(:,1:it);
W = W(:,1:it);
loglike_MCMC = loglike_MCMC_save;
loglike_SMC = loglike_SMC(:,1:it);
logweight = logweight(:,it); %Only last component for MC averages
particles=struct('y',y,'beta',beta,'Sigma', Sigma,'mu',mu, 'tau',tau, 'rho', rho, 'V', V, 'W',W);

end