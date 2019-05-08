function [ratio_prop] =  ty_ratio(y,y_new,l,u)
%ty_ratio: computes the log ratio of derivates of the transformation required
%to compute the acceptance ratio in MH algorithm
% INPUTS:
%  y: current constrained variable (dx1)
%  y_new: proposed constrained variable (dx1)
%  l: lower bound for y (dx1)
%  u: upper bound for y (dx1)
% OUTPUTS: 
%  ratio_prop: log of the ratio of the proposals in the MH acceptance ratio

% Initialise
d = length(y);
ratio_prop = 0;

for i = 1:d
    if ~(isinf(l(i)) && isinf(u(i))) %Ratio evaluated only if y not in R
        if isinf(l(i))
            % Only the upper bound is finite
            ratio_prop = ratio_prop + log(u(i) - y_new(i)) - log(u(i) - y(i));                
        elseif isinf(u(i))
            % Only the lower bound is finite
            ratio_prop = ratio_prop + log(y_new(i) - l(i)) - log(y(i) - l(i));
        else
            % Both bounds are finite
            ratio_prop = ratio_prop + log(y_new(i) - l(i)) + log(u(i) - y_new(i)) - log(y(i) - l(i)) - log(u(i) - y(i));
        end
        % else: y is in R and log-ratio is 0!
    end
end