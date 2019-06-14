function [ zpred,ypred,zmedian,ymedian,fzpred,fypred] = PredictMarginal(xnew,zgrid,ygrid,p,q,particles,logweight)
%PredictMarginal predicts the mean, median, and density of the 
%(undiscretised) z and latent y for a set of new covariates
% INPUTs:
%   xnew: covariate matrix of size nx(p+q)
%   zgrid: grid for evaluatation of the density for each of the age at 
%          event variables, of size lgrid x b
%   ygrid: grid for evaluatation of the density  of the log age at 
%          even, of size lygrid x b
%   p: number of numerical covariates
%   q: number of discrete (binary expanded) covariates
%   particles: structure of particles produced by the MCMC algorithm
%   logweight: log of the unnormalised weights of the particles
% OUTPUTs:
%    zpred: matrix of mean of (undiscretised) z of size n x d
%    ypred: matrix of mean of latent y of size n x d
%    zmedian: matrix of median of (undiscretised) z of size n x b
%    ymedian: matrix of median of latent y of size n x b
%    fzpred: matrix of density at the (undiscretised) z of size nxlgridxb
%    fypred: matrix of density at the latent y of size nxlygridxb

n=size(xnew,1); %number of new covariates
lgrid=length(zgrid); %size of z grid
lygrid=length(ygrid); % size of y grid
S=size(particles.beta,1); %number of particles
K=size(particles.beta,2); %number of components
d=size(particles.beta,4); % dimension of reponse
b=size(zgrid,2); % number of age at event variables

%Normalise the particle weights
nweight = exp(logweight - max(logweight));
nweight = nweight / sum(nweight);

% Initialize
zpred=zeros(n,d);
ypred=zeros(n,d);
fzpred=zeros(n,lgrid,b);
fypred=zeros(n,lygrid,b);
zmedian=zeros(n,b);
ymedian=zeros(n,b);

% add intercept
xnewmat=[ones(n,1),xnew];

% Average predictions across particles
for s=1:S
    
    %Initialise
    normconst_s=zeros(n,1);
    zpred_s=zeros(n,d);
    ypred_s=zeros(n,d);
	fzpred_s=zeros(n,lgrid,b);
	fypred_s=zeros(n,lygrid,b);
    
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
        zpred_s(:,1:b)=zpred_s(:,1:b)+repmat(wx_sk,1,b).*exp(mean_sk(:,1:b)+1/2*repmat(diag(squeeze(particles.Sigma(s,k,1:b,1:b)))',n,1));
        zpred_s(:,(b+1):d)=zpred_s(:,(b+1):d)+repmat(wx_sk,1,d-b).*normcdf(mean_sk(:,(b+1):d)./repmat(sqrt(diag(squeeze(particles.Sigma(s,k,(b+1):d,(b+1):d))))',n,1));
        ypred_s=ypred_s+repmat(wx_sk,1,d).*mean_sk;
        
        %Update density with weighted component specific density
        for id=1:b
            fzpred_s(:,:,id)=fzpred_s(:,:,id)+repmat(wx_sk,1,lgrid)./(sqrt(particles.Sigma(s,k,id,id)*2*pi)*repmat(zgrid(:,id)',n,1)).*exp(-1/(2*particles.Sigma(s,k,id,id))*(log(repmat(zgrid(:,id)',n,1))-repmat(mean_sk(:,id),1,lgrid)).^2);
            fypred_s(:,:,id)=fypred_s(:,:,id)+repmat(wx_sk,1,lygrid)/sqrt(particles.Sigma(s,k,id,id)*2*pi).*exp(-1/(2*particles.Sigma(s,k,id,id))*(repmat(ygrid(:,id)',n,1)-repmat(mean_sk(:,id),1,lygrid)).^2);
        end
        
        %Update normalising constant of weights
        normconst_s=normconst_s+wx_sk;
    end
    
    %Normailse and update with weigted particle predictions
    zpred_s=zpred_s./repmat(normconst_s,1,d);
    zpred=zpred+zpred_s*nweight(s);
    ypred_s=ypred_s./repmat(normconst_s,1,d);
    ypred=ypred+ypred_s*nweight(s);
    for id=1:b
        fzpred_s(:,:,id)=fzpred_s(:,:,id)./repmat(normconst_s,1,lgrid);
        fypred_s(:,:,id)=fypred_s(:,:,id)./repmat(normconst_s,1,lygrid);
    end
    fzpred=fzpred+fzpred_s*nweight(s);
    fypred=fypred+fypred_s*nweight(s);
end

% Compute median from density evaluated on grid
dygrid=ygrid(2)-ygrid(1);
dzgrid=zgrid(2)-zgrid(1);
for id=1:b
    for i=1:n
        ind=1;
        prob=dygrid*fypred(i,ind,id);
        zind=1;
        zprob=dzgrid*fzpred(i,zind,id);
        %find index where probability=0.5
        while(prob<0.5)
            ind=ind+1;
            prob=prob+dygrid*fypred(i,ind,id);
        end
        wy=[prob-0.5,0.5-prob+dygrid*fypred(i,ind,id)];
        wy=wy/sum(wy);
        ymedian(i,id)=wy(2)*ygrid(ind)+wy(1)*ygrid(ind-1);
        %find index where probability=0.5
        while(zprob<0.5)
            zind=zind+1;
            zprob=zprob+dzgrid*fzpred(i,zind,id);
        end
        wz=[zprob-0.5,0.5-zprob+dzgrid*fzpred(i,zind,id)];
        wz=wz/sum(wz);
        zmedian(i,id)=wz(2)*zgrid(zind)+wz(1)*zgrid(zind-1);
    end
end

end

