function [gmm] = gmmCreateCluster(gmm,x_ori,I)
gmm.k = gmm.k + 1;
gmm.weight(gmm.k) = 1;
gmm.TrackerC(gmm.k,:) = x_ori;
gmm.TrackerS(gmm.k,:) = ((0.01*ones(1,I)));
gmm.Np(gmm.k) = 1;
gmm.Ni(gmm.k) = 0;                      % Ni and Nc will be updated in the
gmm.Nc(gmm.k,:) = zeros(1,gmm.nClass);  % next step, after the creation
gmm.accu(gmm.k) = 0;
gmm.counter_accu(gmm.k) = 0;
gmm.weight = gmm.weight./sum(gmm.weight);   % normalize the weight
gmm.gmmwinner = gmm.k;
end

