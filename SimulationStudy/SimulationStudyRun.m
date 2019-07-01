clear all

%Load folder with source codes
addpath('../Codes')

%% Load simulated data
load('simulateddata')
% z: contains the response variables of size nxd, where the first b are 
%    ages at events and d-b are binary
% c: contains the censoring dummies of size nxb
% x: covariate matrix of size nx(p+q), where the first p are numerical and
%    the last q are binary expansions of categorical variables (taking
%    categories 1 or 2!!)
% n: sample size
% p,q: number of numerical and binary expanded categorical variables,
%      respecitvely
d=size(z,2); %dimension of response 
% In the simulated data: d-1 age at event variables and 1 binary variable
b=size(c,2); %number of age at event variables

%% Link functions

% Note: link functions and inverse link functions have been defined in the
% functions linkfunctions.m and invlinkfunctions.m, which assume the first
% b response variables are age at event variables and the last d-b are
% binary. For other input types, these functions need to be changed!

%% MCMC and SMC parameters

% MCMC
burnin = 10000; %burnin period
every = 20; %thinning
start_trunc = 15; %Initial number of components + 1
numbofparts = 1000; %MCMC sample size

% SMC
% SMC Stopping Rule parameters are: epsilon_trunc and numb_trunc, where:
% STOP if |ESS_{J+1}-ESS_J|<epsilon_trunc*numbofparts more than numb_trunc times
epsilon_trunc = 0.01;
numb_trunc = 3;
% number of mcmc samples if rejuvenation is required in the SMC
numbofMCMC = 3; 
% Maximum truncation level
top_trunc = 100;

%Define a struct for mcmc and smc parameters
mcmcsmc = struct('start_trunc', start_trunc, 'numbofparts', numbofparts, 'epsilon_trunc', epsilon_trunc, 'numb_trunc', numb_trunc, 'numbofMCMC', numbofMCMC, 'top_trunc', top_trunc, 'burnin', burnin, 'every', every);

%% Hyperparameters

% Default: empirical priors
gprior=10; %constant in gprior
hyperparameters=empiricalhyperparameters(x(:,1:p),z,c,p,0,gprior);

% Can re-specify any parameters: 
% for multivariate linear regression compoenets: beta0, U_iC, Sigma0, nu
% for continuous x: mu0, ic, a1, a2
% for categorical x: alpha_rho
% for stickbreaking: M
% e.g. hyperparameters.M=2;

%% Run algorithm
[particles, logweight, loglike_MCMC, loglike_SMC, ESS_SMC] = AT_NWR(z, c, x(:,1:p), p, 0, mcmcsmc, hyperparameters);
        
%% Traceplots
S = mcmcsmc.numbofparts;
K = size(loglike_SMC,2);

% All iterations
figure;
plot(loglike_MCMC, 'k', 'LineWidth', 1.5)
xlabel('Iteration')
title('Log-likelihood')

% Saved iterations
figure;
plot(1:numbofparts,loglike_MCMC(burnin+1:every:end), 'k', 'LineWidth', 1.5)
xlabel('Iteration')
title('Log-likelihood')

%% Plot particle weights for SMC
figure
plot(logweight)
%Normalise the particle weights
nweight = exp(logweight - max(logweight));
nweight = nweight / sum(nweight);

%% Estimated (undiscretized) ages for observed data
% Compute expectation of y and exp(y)
yhat=squeeze(sum(particles.y(:,:,1:b).*repmat(nweight,1,n,b)));
zhat=squeeze(sum(exp(particles.y(:,:,1:b)).*repmat(nweight,1,n,b)));
% Compute median of exp(z)
zhat2=zeros(n,b);
for i=1:n
    for id=1:b
        [ysort, isort]=sort(particles.y(:,i,id));
        cnweights=cumsum(nweight(isort'));
        ind=sum(cnweights<.5);
        zhat2(i,id)=exp(ysort(ind));
    end
end

% Plot (undiscretized) ages for observed data
% create indicator for categorial variables
catind=zeros(n,6);
catind(:,1)=(x(:,p+1)==1)&(x(:,p+2)==1)&(x(:,p+3)==1);
catind(:,2)=(x(:,p+1)==2)&(x(:,p+2)==1)&(x(:,p+3)==1);
catind(:,3)=(x(:,p+1)==1)&(x(:,p+2)==2)&(x(:,p+3)==1);
catind(:,4)=(x(:,p+1)==1)&(x(:,p+2)==1)&(x(:,p+3)==2);
catind(:,5)=(x(:,p+1)==2)&(x(:,p+2)==1)&(x(:,p+3)==2);
catind(:,6)=(x(:,p+1)==1)&(x(:,p+2)==2)&(x(:,p+3)==2);
colors=['m','r','g','b','k','c'];
titles_Cat=['Category=(1,1)';'Category=(2,1)';'Category=(1,2)';'Category=(2,2)';'Category=(1,3)';'Category=(2,3)'];
for id=1:b
	figure
    hold on
    for j=1:6
        plot(x(catind(:,j)==1&c(:,id)==1,1),zhat2(catind(:,j)==1&c(:,id)==1,id),'x','Color',colors(j))
    end
    for j=1:6
       plot(x(catind(:,j)==1&c(:,id)==0,1),zhat2(catind(:,j)==1&c(:,id)==0,id),'*','Color',colors(j))
    end
    legend(titles_Cat,'Location','northwest')
    xlabel('x_1')
    ylabel(['z',num2str(id)])
    hold off
end

%% Prediction of Marginals of the d response variables
x1_grid = ((min(x(:,1))):0.2:(max(x(:,1))))'; % grid for age at interview
xnew=repmat(x1_grid,6,1); % 6=2 categories for first discrete covariate * 3 categories for second discrete covariate
xnew=[xnew,repmat(kron([1 1;2 1;1 2],ones(length(x1_grid),1)),2,1)]; 
xnew=[xnew, kron((1:2)',ones(3*length(x1_grid),1))]; 
z1_grid = [((min(z(:,1))-4):.2:(max(z(:,1))+10))]; % grid for age at event
zgrid=[z1_grid' z1_grid'];
y1_grid=(log(min(z(:,1))-4):.05:log(max(z(:,1))+10)); % grid for log of age at event
ygrid=[y1_grid' y1_grid'];

% Call function to predict marginals (mean, median, and density)
[zmean,ymean,zmedian,ymedian,fz,fy]=PredictMarginal(xnew,zgrid,ygrid,p,q,particles,logweight);

% Plot Means
catnewind=zeros(size(xnew,1),6);
catnewind(:,1)=(xnew(:,p+1)==1)&(xnew(:,p+2)==1)&(xnew(:,p+3)==1);
catnewind(:,2)=(xnew(:,p+1)==2)&(xnew(:,p+2)==1)&(xnew(:,p+3)==1);
catnewind(:,3)=(xnew(:,p+1)==1)&(xnew(:,p+2)==2)&(xnew(:,p+3)==1);
catnewind(:,4)=(xnew(:,p+1)==1)&(xnew(:,p+2)==1)&(xnew(:,p+3)==2);
catnewind(:,5)=(xnew(:,p+1)==2)&(xnew(:,p+2)==1)&(xnew(:,p+3)==2);
catnewind(:,6)=(xnew(:,p+1)==1)&(xnew(:,p+2)==2)&(xnew(:,p+3)==2);
for id=1:d
	figure
    hold on
    for j=1:6
        plot(xnew(catnewind(:,j)==1,1),zmean(catnewind(:,j)==1,id),'-','Color',colors(j),'LineWidth',2)
    end
    if id<=b
        for j=1:6
            plot(x(catind(:,j)==1&c(:,id)==1,1),zhat2(catind(:,j)==1&c(:,id)==1,id),'x','Color',colors(j))
            plot(x(catind(:,j)==1&c(:,id)==0,1),zhat2(catind(:,j)==1&c(:,id)==0,id),'*','Color',colors(j))
        end
    end
    legend(titles_Cat,'Location','northwest')
    xlabel('x_1')
    ylabel(['z',num2str(id)])
    hold off
end

% Plot Medians
for id=1:b
	figure
    hold on
    for j=1:6
        plot(xnew(catnewind(:,j)==1,1),zmedian(catnewind(:,j)==1,id),'-','Color',colors(j),'LineWidth',2)
    end
    for j=1:6
        plot(x(catind(:,j)==1&c(:,id)==1,1),zhat2(catind(:,j)==1&c(:,id)==1,id),'x','Color',colors(j))
        plot(x(catind(:,j)==1&c(:,id)==0,1),zhat2(catind(:,j)==1&c(:,id)==0,id),'*','Color',colors(j))
    end
    legend(titles_Cat,'Location','northwest')
    xlabel('x_1')
    ylabel(['z',num2str(id)])
    hold off
end

% Plot heat maps of the density estimates
mycolors=mycolormap(20,1); %color map for heat map
titlez=['$Z_1$ for ';'$Z_2$ for '];
for id=1:b
    figure
    for j=1:6
    subplot(2,3,j)
    colormap(mycolors)
    % density heat map
    imagesc(x1_grid, zgrid(:,id)',fz(catnewind(:,j)==1,:,id)')
    set(gca,'Ydir','normal');
    hold on
    % add median curve
    plot(x1_grid,zmedian(catnewind(:,j)==1,id),'k','LineWidth',2)
    % add observed data
    plot(x(catind(:,j)==1&c(:,id)==1,1),z(catind(:,j)==1&c(:,id)==1,id),'kx')
    % add censored data with median of Z
    plot(x(catind(:,j)==1&c(:,id)==0,1),zhat2(catind(:,j)==1&c(:,id)==0,id),'k*')    
    ylim([10,28])
    title([titlez(id,:),titles_Cat(j,:)],'Interpreter','latex','Fontsize',14)
    hold off
    end
end

%% Censoring Probability, Surival, and Hazard functions
% Call function to predict censoring probability and survival curves
[censorprob,Sz] = PredictCensorSurvival(xnew,zgrid,p,q,particles,logweight);
%Hazard
hz=fz./Sz;

% Plot Censoring Probability
titlez2=['$Z_1$';'$Z_2$'];
for id=1:b
	figure
    hold on
    for j=1:6
        plot(xnew(catnewind(:,j)==1,1),censorprob(catnewind(:,j)==1,id),'-','Color',colors(j),'LineWidth',2)
    end
    legend(titles_Cat,'Location','northwest')
    xlabel('x_1')
    title(['Censoring Probability of ',titlez2(id,:)],'Interpreter','latex','Fontsize',14)
    hold off
end

% Plot Surival curve
xtoplot=[15,18,21,24,27,29];
for id=1:b
	figure
    for xid=1:length(xtoplot)
        subplot(2,3,xid)
        hold on
        for j=1:6
            plot(zgrid(:,id),Sz(catnewind(:,j)==1&xnew(:,1)==xtoplot(xid),:,id)','-','Color',colors(j),'LineWidth',2)
        end
        legend(titles_Cat,'Location','northeast')
        xlabel(titlez2(id,:),'Interpreter','latex')
        title(['Survival for Age at Interview =', num2str(xtoplot(xid))],'Fontsize',14)
        hold off
    end
end

% Plot Hazard function
xtoplot=[15,18,21,24,27,29];
for id=1:b
	figure
    for xid=1:length(xtoplot)
        subplot(2,3,xid)
        hold on
        for j=1:6
            plot(zgrid(:,id),hz(catnewind(:,j)==1&xnew(:,1)==xtoplot(xid),:,id)','-','Color',colors(j),'LineWidth',2)
        end
        legend(titles_Cat,'Location','northwest')
        xlabel(titlez2(id,:),'Interpreter','latex')
        title(['Hazard for Age at Interview =', num2str(xtoplot(xid))],'Fontsize',14)
        hold off
    end
end

%% Conditionals of Age at Event variables
x1c = [15,20,25,29]'; % age at interview (x1) for conditional prediction
xnewc=repmat(x1c,6,1);  % 6=2 categories for first discrete covariate * 3 categories for second discrete covariate
xnewc=[xnewc,repmat(kron([1 1;2 1;1 2],ones(length(x1c),1)),2,1)]; 
xnewc=[xnewc, kron((1:2)',ones(3*length(x1c),1))]; 
id1=2; %indicator for response variable
id2=1; %indicator for conditioned variable
z1gridc = zgrid(:,id1); % grid for response variable
z2gridc=(min(z(:,1)):.2:max(z(:,1))); % grid for conditioned variable
[zcmean, zcmedian,zcdens]=PredictConditional(xnewc,z1gridc,z2gridc,id1,id2,p,q,b,particles,logweight);

% Plot Conditional Means
catnewindc=zeros(size(xnewc,1),6);
catnewindc(:,1)=(xnewc(:,p+1)==1)&(xnewc(:,p+2)==1)&(xnewc(:,p+3)==1);
catnewindc(:,2)=(xnewc(:,p+1)==2)&(xnewc(:,p+2)==1)&(xnewc(:,p+3)==1);
catnewindc(:,3)=(xnewc(:,p+1)==1)&(xnewc(:,p+2)==2)&(xnewc(:,p+3)==1);
catnewindc(:,4)=(xnewc(:,p+1)==1)&(xnewc(:,p+2)==1)&(xnewc(:,p+3)==2);
catnewindc(:,5)=(xnewc(:,p+1)==2)&(xnewc(:,p+2)==1)&(xnewc(:,p+3)==2);
catnewindc(:,6)=(xnewc(:,p+1)==1)&(xnewc(:,p+2)==2)&(xnewc(:,p+3)==2);
for idx=1:length(x1c)
	figure
    hold on
    for j=1:6
        plot(z2gridc,zcmean(catnewindc(:,j)==1&xnewc(:,1)==x1c(idx),:),'-','Color',colors(j),'LineWidth',2)
    end
    % Add observed data
    for j=1:6
        % Crosses if noncensored for both responses
        plot(z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id2),z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id1),'x','Color',colors(j))
        % Stars if censored for either response
        plot(zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id2),zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id1),'*','Color',colors(j))
    end
    legend(titles_Cat,'Location','northwest')
    xlabel(['z',num2str(id2)])
    ylabel(['z',num2str(id1)])
    hold off
end

% Plot Conditional Medians
for idx=1:length(x1c)
	figure
    hold on
    for j=1:6
        plot(z2gridc,zcmedian(catnewindc(:,j)==1&xnewc(:,1)==x1c(idx),:),'-','Color',colors(j),'LineWidth',2)
    end
    % Add observed data
    for j=1:6
        % Crosses if noncensored for both responses
        plot(z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id2),z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id1),'x','Color',colors(j))
        % Stars if censored for either response
        plot(zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id2),zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id1),'*','Color',colors(j))
    end
    legend(titles_Cat,'Location','northwest')
    xlabel(['z',num2str(id2)])
    ylabel(['z',num2str(id1)])
    hold off
end

% Plot heat maps of the conditional density estimates
mycolors=mycolormap(20,1); %color map for heat map
for idx=1:length(x1c)
    figure
    for j=1:6
    subplot(2,3,j)
    colormap(mycolors)
    % density heat map
    imagesc(z2gridc, z1gridc,squeeze(zcdens(catnewindc(:,j)==1&xnewc(:,1)==x1c(idx),:,:))')
    set(gca,'Ydir','normal');
    hold on
    % add median curve
    plot(z2gridc,zcmedian(catnewindc(:,j)==1&xnewc(:,1)==x1c(idx),:),'k-','LineWidth',2)
    % add observed data
    plot(z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id2),z(catind(:,j)==1&c(:,id1)==1&c(:,id2)==1&x(:,1)==x1c(idx),id1),'kx')
    % add censored data (in either response) with median of Z
    plot(zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id2),zhat2(catind(:,j)==1&(c(:,id1)==0|c(:,id2)==0)&x(:,1)==x1c(idx),id1),'k*')
    ylim([10,40])
    xlabel(['z',num2str(id2)])
    ylabel(['Density z',num2str(id1)])
    title(titles_Cat(j,:),'Interpreter','latex','Fontsize',14)
    hold off
    end
end
