function [z,c] = linkfunctions(y,x,d,b)
%linkfunctions defines functions mapping latent y to observed response z
% INPUTs:
%   x: covariate matrix of size nx(p+q): first column is age at interview
%   y: latent response of size nxd 
%   d: dimension of response
%   b: number of age at event variables
% OUTPUTs:
%   z: observed response of size nxd
%   c: censoring dummies of size nxb

% Initialise
n=size(y,1);
z=zeros(n,d);
c=zeros(n,b);

% Assume first b are age at event variables
for id=1:b
    zaux=floor(exp(y(:,id)));
    z(zaux<(x(:,1)+1),id)=zaux(zaux<(x(:,1)+1));
    z(zaux>=(x(:,1)+1),id)=NaN;
    c(zaux<(x(:,1)+1),id)=1;
end
% Assume last d-b are binary variables
for id=(b+1):d
    z(:,id)=y(:,id)>0;
end

