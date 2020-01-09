function gmm = gmmInitialize(nInput,nClass,center_init,mode)
gmm.k               = 1;                           %no of cluster
gmm.nClass          = nClass;                      %no of classes
gmm.counter         = 1;                           %no of data point
gmm.weight          = 1;                           %mixture weight
gmm.TrackerC        = center_init;                 %center of every clusters
gmm.TrackerS        = ((0.01*ones(1,nInput)));     %variance
gmm.Np              = 1;                           %the number of times a particular cluster has won the competition
gmm.Ni              = 1;                           %the number of classes assigned in a cluster
gmm.Nc              = zeros(1,nClass);             %the number of particular classes assigned in a cluster
gmm.Nc(1)           = 1;
gmm.accu            = 0;                           %accumulator of cluster activity
gmm.counter_accu    = 0;                           %the age of a cluster
gmm.rho             = 0.1;                         %cluster overlap degree
gmm.calculation     = 1;                           %turn on or off the GMM
gmm.gmmwinner       = 1;

% the initial number of clusters is the same with the number of classes
if mode(5) == 1
    for iClass = 1:nClass-1
        smallPerturbation = center_init + normrnd(0,0.001);
        gmm = gmmCreateCluster(gmm,smallPerturbation,nInput);
        gmm.Nc(gmm.k,iClass+1) = 1;
        gmm.Ni(gmm.k) = 1;
    end
end
end

