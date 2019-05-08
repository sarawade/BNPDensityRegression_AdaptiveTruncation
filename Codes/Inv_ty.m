function [y] = Inv_ty(ty, l, u)
%Inv_ty: inverse of ty transformation to obtain constrain y in (l,u) from
%real-valued t.
% INPUTS
% ty: real valued transformed variable (dx1)
% l: lower bound for y (dx1)
% u: upper bound for y (dx1)
% OUTPUTS
% y: constrained variable (dx1)

%Initailise
y=zeros(size(l));
d=max(size(l));

for i=1:d
    if (isinf(l(i)) && isinf(u(i)))
        %Both bounds are infinite, No transformation needed
            y(i)=ty(i);
    elseif (isinf(l(i)) && ~isinf(u(i)))
        % Only upper bound is finite
        y(i) = u(i) - exp(-ty(i));
    elseif (~isinf(l(i)) && isinf(u(i)))
        % Only lower bound is finite
        y(i) = l(i) + exp(ty(i));
    else
        % Both bounds are finite
        if ty(i)<=0
            % More stable computation if t<=0
            y(i) = (l(i) + u(i).*exp(ty(i)))./(1 + exp(ty(i)));
            % Correct if computational error
            if y(i)<l(i)
                y(i)=l(i); 
            end
        else
            % More stable computation if t>0
            y(i) = (u(i) + l(i).*exp(-ty(i)))./(1 + exp(-ty(i)));
            % Correct if computational error
            if y(i)>u(i)
                y(i)=u(i);
            end
        end
    end
end
end