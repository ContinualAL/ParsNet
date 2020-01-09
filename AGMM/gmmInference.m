function [Vn,expn] = gmmInference(gmm,x_ori)
Vn = [];        % volume
expn = [];      % activity
nGmm = gmm.k;   % number of GMM

for iGmm = 1:nGmm
    inverseVariance = (1./gmm.TrackerS(iGmm,:));        % 1/sigma
    clusterCentroid = (gmm.TrackerC(iGmm,:));
    dist = (x_ori - clusterCentroid).*inverseVariance.*...
        (x_ori - clusterCentroid);                      
    [inference,maxDistance] = min(exp(-0.5*dist));
    expn(iGmm) = inference;                             % gmm inference eqn 12
    Vn(iGmm) = gmm.TrackerS(iGmm,maxDistance);          % volume
end
end