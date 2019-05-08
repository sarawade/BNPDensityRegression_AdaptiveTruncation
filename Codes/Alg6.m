function [s_d,sum_param,prod_param,param_sd] = Alg6(param, sum_param, prod_param, param_sd, s_d, iter, ADAPT)
%Alg6: adaptive update of the variance of the proposal based on
%Algorithm 6 of Griffin and Stephens (2013)
% INPUTS:
%  param:proposed value of parameter block (px1)
%  sum_param: cummulative sum of parameters based on previous iter-1
%             iterations (px1)
%  prod_param: cummulative sum of the product of parameters based on 
%              previous iter-1 iterations (pxp)
%  param_sd: current value of adaptive variance (only used to set updated
%            variance when the number of iterations < ADAPT(1)) (pxp)
%  s_d: current value of scaling factor (1x1)
%  iter: number of MCMC iterations (1x1)
%  ADAPT: vector of five elements containing 1) number of iterations to
%         start adaptation, 2) parameter determining update of scaling
%         factor, 3) acceptance probability of proposal, 4) target
%         acceptance probability, 5) nugget determining min level of
%         exploration and ensuring invertability
% OUTPUTS:
%  s_d: updated value of scaling factor (1x1)
%  sum_param: updated cummulative sum of parameters based on iter
%             iterations (px1)
%  prod_param: updated cummulative sum of the product of parameters based  
%              on iter iterations (pxp)
%  param_sd: updated value of adaptive variance 

% size of parameter block
p = size(param,1);

%Update the scaling factor based on ADAPT 
s_d = s_d * exp(iter^(-ADAPT(2))*(ADAPT(3) - ADAPT(4)));
if ( s_d > exp(50) )
    s_d = exp(50);
end
if ( s_d < exp(-50) )
    s_d = exp(-50);
end

% Compute cummulative sum and product
sum_param = sum_param + param;
prod_param = prod_param + param*param';

% Only update variance after ADAPT(1) iterations
if iter > ADAPT(1)
    param_sd = s_d/(iter-1)*(prod_param-sum_param*sum_param'/iter) + s_d*ADAPT(5)*eye(p);
end