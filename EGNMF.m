
function [U_final, V_final, nIter_final, objhistory_final] = EGNMF(X, k, W,options, U, V, M)
% Error Graph Regularized Non-negative Matrix Factorization(EGNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%               options.alpha ... the regularization parameter. 
%                                 [default: 100]
%                                 alpha = 0, GNMF boils down to the ordinary NMF. 
%                                 
% 
% You only need to provide the above four inputs.
%
% X = U*V'
%
%
%   Written by Deng Cai (dengcai AT gmail.com)
%   Partial addition modification  by mei-jun

if min(min(X)) < 0
    error('Input should be nonnegative!');
end

if ~isfield(options,'error')
    options.error = 1e-5;
end

if ~isfield(options, 'maxIter')
    options.maxIter = [];
end

if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end

if ~isfield(options,'minIter')
    options.minIter = 30;
end

if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end

if ~isfield(options,'alpha')
    options.alpha = 100;
end

if ~isfield(options,'as')
    options.as = 10;
end


nSmp = size(X,2);

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    options.alpha = options.alpha*nSmp;    
end



if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    feaSum = full(sum(X,2));
    D_half = X'*feaSum;
    X = X*spdiags(D_half.^-.5,0,nSmp,nSmp);
end

if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end

if ~exist('U','var')
    U = [];
    V = [];
end
switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, V_final, nIter_final, objhistory_final] =EGNMF_Multi(X, k, W,options, U, V);
    otherwise
        error('optimization method does not exist!');
end