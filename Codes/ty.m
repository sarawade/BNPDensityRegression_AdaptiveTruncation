function [t] = ty(y, l, u)
%ty: transforms y in (l,u) to real line.
% INPUTS
% y: constrained variable (dx1)
% l: lower bound for y (dx1)
% u: upper bound for y (dx1)
% OUTPUTS
% t: transformed y (dx1)

% Initialise
t=zeros(size(l));
d=max(size(l));

for i = 1:d
    if (isinf(l(i)) && isinf(u(i)))
        %Both bounds are infinite, No transformation needed
        t(i) = y(i);
    elseif (isinf(l(i)) && ~isinf(u(i)))
        % Only upper bound is finite
        t(i) = -log(u(i) - y(i));
    elseif (~isinf(l(i)) && isinf(u(i)))
        % Only lower bound is finite
        t(i) = log(y(i)-l(i));
    else
        % Both bounds are finite
        t(i)=log((y(i)-l(i))/(u(i)-y(i)));
    end
end
end