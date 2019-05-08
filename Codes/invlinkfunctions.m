function [lower,upper] = invlinkfunctions(z,c,x)
%invlinkfunctions defines the lower and uppler bounds for the latent y 
%                 based on inverting the link functions mapping latent y to
%                 observed response z, defined in linkfunctions
% INPUTs:
%   x: covariate matrix of size nx(p+q): first column is age at interview
%   z: observed response response of size nxd 
%   c: censoring dummies of size nxb
% OUTPUTs:
%   lower: lower bounds for latent y of size nxd
%   upper: upper bounds for latent y of size nxb

% Initialise
n=size(z,1); % sample size
d=size(z,2); % dimension of response
b=size(c,2); % dimension of age at event response
lower=zeros(n,d);
upper=zeros(n,d);

for id=1:d
    if id<=b
        % Age at event response
        % for censored: lower bound is log(Age at Interview+1)
        lower(c(:,id) == 0,id) = log(x(c(:,id) == 0,1)+1);
        % for censored: upper bound is Inf
        upper(c(:,id) == 0,id) = Inf;
        % for uncensored: lower bound is log(Age at Event)
        lower(c(:,id) == 1,id) = log(z(c(:,id) == 1,id));
        % for uncensored: upper bound is log(Age at Event+1)
        upper(c(:,id) == 1,id) = log(z(c(:,id) == 1,id) + 1);
    else
        % Binary response: 
        % if z=0 
        lower(z(:,id) == 0,id) = -Inf;
        % if z=1
        upper(z(:,id) == 1,id) = Inf;  
    end
end
end

