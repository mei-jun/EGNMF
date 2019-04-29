function [U_final, V_final, nIter_final, objhistory_final] = GNMF_Multi_W(X, k, W,options, U, V)
%EGNMF with multiplicative update
% A Novel Graph Regularized Non-negative Matrix Factorization based 
% Error Weight Matrix for High Dimensional Biomedical Data Clustering         
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
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  
%

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
options.WeightMode = 'Binary';  
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;
as = options.as;

Norm = 2;
NormV = 0;

[mFea,nSmp]=size(X);

selectInit = 1;
if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end


WW=W;

if alpha > 0
    W = alpha*W;
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
    if isfield(options,'NormW') && options.NormW
        D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
        L = D_mhalf*L*D_mhalf;
        disp(L)
    end
else
    L = [];
end


% Error Weight Matrix
W1 = constructW(V,options);
    W2=abs(W1-WW)./(nSmp*nSmp);
    
    if as > 0
    W2 = as*W2;
    DCol = full(sum(W2,2));
    D2 = spdiags(DCol,0,nSmp,nSmp);
    L2 = D2 - W2;
    if isfield(options,'NormW') && options.NormW
        D_mhalff = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
        L2 = D_mhalff*L*D_mhalff;
        disp(L2)
    end
   else
    L2 = [];
    end
    
    

[U,V] = NormalizeUV(U, V, NormV, Norm);
if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, L,L2);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, L,L2);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end



tryNo = 0;
nIter = 0;
while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;

    while(maxErr > differror)
        % ===================== update V ========================
        XU = X'*U;  
        UU = U'*U;  
        VUU = V*UU; 
        WV = W*V;
        DV = D*V;     
        WV2 = W2*V;
        DV2 = D2*V;   
        XU = XU + WV+WV2;
        VUU = VUU + DV+DV2;
        V = V.*(XU./max(VUU,1e-10));
        
        % ===================== update U ========================
        XV = X*V;   % mnk or pk (p<<mn)
        VV = V'*V;  % nk^2
        UVV = U*VV; % mk^2
        
        U = U.*(XV./max(UVV,1e-10)); % 3mk
        
    W1 = constructW(V,options);
    W2=abs(W1-WW)./(nSmp*nSmp);
    
    if as > 0
    W2 = as*W2;
    DCol = full(sum(W2,2));
    D2 = spdiags(DCol,0,nSmp,nSmp);
    L2 = D2 - W2;
    if isfield(options,'NormW') && options.NormW
        D_mhalff = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
        L2 = D_mhalff*L*D_mhalff;
        disp(L2)
    end
   else
    L2 = [];
    end
        
  
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, L,L2);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, L,L2);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, L,L2);
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(U, V, NormV, Norm);
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);


%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, L,L2, deltaVU, dVordU)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U + L*V + L2*V;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V+ L2*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((V'*L).*V'))+sum(sum((V'*L2).*V'));
    end
    obj = obj_NMF+obj_Lap;
    




function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end

        