function [censorprob,Szpred] = PredictCensorSurvival(xnew,zgrid,p,q,particles,logweight)
%PredictCensorSurvival predicts the censoring probability and survival
%curves at (undiscretized) z for a set of new covariates
% INPUTs:
%   xnew: covariate matrix of size nx(p+q)
%   zgrid: grid for evaluatation of the survival curve for each age at 
%          event variables, of size lgrid x b
%   p: number of numerical covariates
%   q: number of discrete (binary expanded) covariates
%   particles: structure of particles produced by the MCMC algorithm
%   logweight: log of the unnormalised weights of the particles
% OUTPUTs:
%    censorprob: matrix of probability that the age at event is bigger than 
%                age at interview (x_1) of size n x b
%    Szpred: matrix of survival curves at the (undiscretised) z for each 
%            new covariate of size nxlgridxb

n=size(xnew,1); %number of new covariates
lgrid=size(zgrid,1);%size of z grid
S=size(particles.beta,1); %number of particles
K=size(particles.beta,2); %number of components
b=size(zgrid,2); % number of age at event variables

%Normalise the particle weights
nweight = exp(logweight - max(logweight));
nweight = nweight / sum(nweight);

% Initialize
Szpred=zeros(n,lgrid,b);
censorprob=zeros(n,b);

% add intercept
xnewmat=[ones(n,1),xnew];

% Average predictions across particles
for s=1:S
    
    %Initialise
    normconst_s=zeros(n,1);
    Szpred_s=zeros(n,lgrid,b);
    censorprob_s=zeros(n,b);
    
    % Average across components
    for k=1:K
        %Compute unnormalised weights
        wx_sk=particles.W(s,k)*ones(n,1);
        if p>0
            for j=1:p
                wx_sk=wx_sk.*normpdf(xnew(:,j),particles.mu(s,k,j),particles.tau(s,k,j)^(-.5));
            end
        end
        if q>0
            for j=1:q
                wx_sk=wx_sk.*(particles.rho(j,s,k).^(xnew(:,p+j)==1)).*((1-particles.rho(j,s,k)).^(xnew(:,p+j)==2));
            end
        end
        
        %Compute component specific mean
        mean_sk=xnewmat*squeeze(particles.beta(s,k,:,:));
        
        %Update predictions with weighted component specific mean
        for id=1:b
            % Compute censor probability
            aux_nocensor_s=(normcdf((log(xnew(:,1))-mean_sk(:,id))/sqrt(particles.Sigma(s,k,id,id))));
            censorprob_s(:,id)=censorprob_s(:,id)+wx_sk.*(1-aux_nocensor_s);
            % Compute survival
            aux_survival_s=normcdf((repmat(log(zgrid(:,id)'),n,1)-repmat(mean_sk(:,id),1,lgrid))/sqrt(particles.Sigma(s,k,id,id)));
            Szpred_s(:,:,id)=Szpred_s(:,:,id)+repmat(wx_sk,1,lgrid).*(1-aux_survival_s);
        end
        
        %Update normalising constant of weights
        normconst_s=normconst_s+wx_sk;
    end
    
    %Normailse and update with weigted particle predictions
    censorprob_s=censorprob_s./repmat(normconst_s,1,b);
    censorprob=censorprob+censorprob_s*nweight(s);
    for id=1:b
        Szpred_s(:,:,id)=Szpred_s(:,:,id)./repmat(normconst_s,1,lgrid);
    end
    Szpred=Szpred+Szpred_s*nweight(s);
end


end

