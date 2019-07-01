function C = mycolormap(N,rev)
%   mycolormap(N,rev) returns an N-by-3 matrix containing a colormap. 
%   The colors begin with blue, range
%   through orange and end with red. The option rev reverses the colormap.
%  
  
mycolors=load('mycolors');
mycolors=mycolors.mycolors;

P = size(mycolors,1);
if nargin < 1
   N = P;
end
if nargin < 2
    rev = 0;
end
if rev==1
    mycolors=mycolors(P:-1:1,:);
end

N = min(N,P);
C = interp1(1:P, mycolors, linspace(1,P,N), 'linear');    