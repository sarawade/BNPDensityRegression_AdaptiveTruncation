function [zmean,zmedian,zdens] = PredictConditional(xnew,z1grid, z2grid,id1,id2,p,q,b,particles,logweight)
%PredictConditional predicts the mean, median, and density of response with
%index id1 conditioned on another response with index id2<=b for a set of new 
%covariates
% INPUTs:
%   xnew: covariate matrix of size nx(p+q)
%   z1grid: grid for evaluatation of the density for the response with 
%           index id1, of size lgrid x 1
%   z2grid: grid of values for the conditioned response variable with 
%           index id2 (with id2<=b for an age at event variable), 
%           of size lgrid x 1
%   id1: indexes response variable to predict
%   id2: indexes conditioned response variable  (id2<=b)
%   p: number of numerical covariates
%   q: number of discrete (binary expanded) covariates
%   b: number of age at event responses
%   particles: structure of particles produced by the MCMC algorithm
%   logweight: log of the unnormalised weights of the particles
% OUTPUTs:
%    zmean: matrix of the conditional mean of response with index id1
%           conditioned on the response with index id2 and the new 
%           covariates, of size n x length(z2grid)
%    zmedian: matrix of the conditional median of response with index id1
%           conditioned on the response with index id2 and the new 
%           covariates, of size n x length(z2grid)
%    zdens: matrix of the conditional density evaluated at the 
%           (undiscretised) z1grid values conditioned on the response
%           with index id2 and the new covariates, 
%           of size n x length(z2grid) x length(z1grid)

n=size(xnew,1); %number of new covariates
lgrid1=length(z1grid); %length of response grid
lgrid2=length(z2grid); %length of conditioned grid
S=size(particles.beta,1); %number of particles
K=size(particles.beta,2); %number of components

%Normalise the particle weights
nweight = exp(logweight - max(logweight));
nweight = nweight / sum(nweight);

% Initialize
zmean=zeros(n,lgrid2);
zdens=zeros(n,lgrid2, lgrid1);
zmedian=zeros(n,lgrid2);
fydens=zeros(n,lgrid2); %need to compute the marginal density of the conditioned variable

% add intercept
xnewmat=[ones(n,1),xnew];

% Average predictions across particles
for s=1:S
    
    %Initialise
    normconst_s=zeros(n,1);
    zpred_s=zeros(n,lgrid2);
    zdens_s=zeros(n,lgrid2, lgrid1);
    fydens_s=zeros(n,lgrid2);
    
    % Average across components
    for k=1:K
        
        %Compute unnormalised weights
        wx_sk=particles.W(s,k)*ones(n,1);
        for j=1:p
            wx_sk=wx_sk.*normpdf(xnew(:,j),particles.mu(s,k,j),particles.tau(s,k,j)^(-.5));
        end
        if q>0
            for j=1:q
                wx_sk=wx_sk.*(particles.rho(j,s,k).^(xnew(:,p+j)==1)).*((1-particles.rho(j,s,k)).^(xnew(:,p+j)==2));
            end
        end
        
        %Compute component specific mean
        mean_sk=xnewmat*squeeze(particles.beta(s,k,:,:));
        
        %Compute conditioned mean and variance
        cmean_sk=repmat(mean_sk(:,id1),1,lgrid2)+particles.Sigma(s,k,id1,id2)/particles.Sigma(s,k,id2,id2)*(repmat(log(z2grid),n,1)-repmat(mean_sk(:,id2),1,lgrid2));
        cvar_sk=particles.Sigma(s,k,id1,id1)-particles.Sigma(s,k,id1,id2)^2/particles.Sigma(s,k,id2,id2);

        %Compute marginal density of conditioned variable
        fydens_sk=normpdf(repmat(log(z2grid),n,1),repmat(mean_sk(:,id2),1,lgrid2),sqrt(particles.Sigma(s,k,id2,id2)));
        fydens_s=fydens_s+repmat(wx_sk,1,lgrid2).*fydens_sk;
        
        %Update predictions with weighted component specific conditional
        %mean and density
        if id1<=b
            %If id1 indexes an age at event variable
            zpred_s=zpred_s+repmat(wx_sk,1,lgrid2).*exp(cmean_sk+1/2*cvar_sk).*fydens_sk;
            for i=1:lgrid1
                zdens_s(:,:,i)=zdens_s(:,:,i)+repmat(wx_sk,1,lgrid2)./(sqrt(cvar_sk*2*pi)*z1grid(i)).*exp(-1/(2*cvar_sk)*(log(z1grid(i))-cmean_sk).^2).*fydens_sk;
            end
        else
            %If id1 indexes a binary variable
            zpred_s=zpred_s+repmat(wx_sk,1,lgrid2).*normcdf(cmean_sk/sqrt(cvar_sk)).*fydens_sk;
        end
        
        %Update normalising constant of weights
        normconst_s=normconst_s+wx_sk;
    end
    
    %Normailse and update with weigted particle predictions
    fydens_s=fydens_s./repmat(normconst_s,1,lgrid2);
    fydens=fydens+fydens_s*nweight(s);
    zpred_s=zpred_s./repmat(normconst_s,1,lgrid2);
    zmean=zmean+zpred_s*nweight(s);
    if id1<=b
        %only compute density if id1 indexes an age at event variable
        zdens_s=zdens_s./repmat(normconst_s,1,lgrid2,lgrid1);
        zdens=zdens+zdens_s*nweight(s);
    end
end
zmean=zmean./fydens;
if id1<=b
    %only compute density if id1 indexes an age at event variable
    for i=1:lgrid1
        zdens(:,:,i)=zdens(:,:,i)./fydens;
    end
end

%Compute median based on density
if id1<=b
    %%only compute median if id1 indexes an age at event variable
    dz1grid=z1grid(2)-z1grid(1);
    for i=1:n
        for j=1:lgrid2
            zind=1;
            zprob=dz1grid*zdens(i,j,zind);
            while(zprob<0.5)
                zind=zind+1;
                zprob=zprob+dz1grid*zdens(i,j,zind);
            end
            wz=[zprob-0.5,0.5-zprob+dz1grid*zdens(i,j,zind)];
            wz=wz/sum(wz);
            zmedian(i,j)=wz(2)*z1grid(zind)+wz(1)*z1grid(zind-1);
        end
    end
end

end

