% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

function [additionalInput,additionalTarget,trueLabel] = selfLabelling(gmm,...
    input,output,trueLabel)
nData = size(output,1);
additionalTarget = [];
indicesCandidate = [];
countData = 0;
for iData = 1:nData
    %% confidenceLevel1: from the network
    currOutput = output(iData,:);
    [~,candidateLabel1] = max(currOutput);
    confCandidate = sort(currOutput,'descend');
    y1 = confCandidate(1);
    y2 = confCandidate(2);
    confidenceLevel1 = y1/(y1+y2);
    
    %% confidenceLevel2: from the AGMM
    currInput = input(iData,:);
    [Vr,expr] = gmmInference(gmm,currInput);
    if sum(expr) == 0
        [~,maxCluster] = max(Vr);
        expr(maxCluster) = expr(maxCluster) + 0.000001;
    end
    denumerator = (2*pi)^(-1/2)*Vr.^(-1/2);                 
    
    pxr = (denumerator + 0.000001).*expr;                   % p(X|Ny)
    pr = gmm.Ni./sum(gmm.Ni);                               % p(Ny)
    denumerator2 = 1./sum(gmm.Nc,2);                        
    pyr = gmm.Nc.*denumerator2;                             % p(Y|Ny)
    denumerator3 = sum(sum(pyr.*(pxr.*pr)'));
    numerator = sum(pyr.*(pxr.*pr)',1);
    pyx = numerator./(denumerator3);                        % p(Y|X)
    
    [~,candidateLabel2] = max(pyx);
    confCandidate2 = sort(pyx,'descend');
    y12 = confCandidate2(1);
    y22 = confCandidate2(2);
    confidenceLevel2 = y12/(y12+y22);
    if candidateLabel1 == candidateLabel2 &&...
            confidenceLevel1 > 0.6 && confidenceLevel2 > 0.55
        countData = countData + 1;
        indicesCandidate = [indicesCandidate iData];
        candidateTarget = zeros(1,gmm.nClass);
        candidateTarget(candidateLabel1) = 1;               % self labeling
        additionalTarget(countData,:) = candidateTarget;
    end
end
additionalInput = input(indicesCandidate,:);
trueLabel = trueLabel(indicesCandidate,:);
end

