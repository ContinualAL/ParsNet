function rho = gmmClusterAvailability(gmm,gmmWinner)
nGmm = gmm.k;

sigmaWinner = sqrt(gmm.TrackerS(gmmWinner,:));
miu_plus_sigma_winner = gmm.TrackerC(gmmWinner,:) + sigmaWinner;
miu_mins_sigma_winner = gmm.TrackerC(gmmWinner,:) - sigmaWinner;

totalProportion = length(gmm.TrackerC(gmmWinner,:))*(gmm.k-1);
numInside = 0;

% count how many centers inside the winning cluster
for iGmm = 1:nGmm
    if iGmm ~= gmmWinner
        center = gmm.TrackerC(iGmm,:);
        lessThanSigma = center < miu_plus_sigma_winner;
        moreThanSigma = center > miu_mins_sigma_winner;
        insideWinner = lessThanSigma.*moreThanSigma;
        numInside = numInside + sum(insideWinner);
    end
end

numOutside = abs(totalProportion-numInside);
rho = numOutside/totalProportion;       % eqn 14
if rho == 0
    rho = 0.1; % minimum vigilance parameter
end
end