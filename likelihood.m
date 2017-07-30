function [ lhood ] = likelihood(X, agmm )
% Computes the likelihood of a data coming from a particular distribution
%   INPUTS : data vector X, GMM Model object agmm
%   OUTPUTS : the P(x/O) (likelihood value) lhood

obj = gmdistribution(agmm.mu',reshape(agmm.sigma,[1 size(agmm.sigma)]),agmm.w');
temp = cell2mat(X);
P = pdf(obj,temp');
lhood = P;

end

