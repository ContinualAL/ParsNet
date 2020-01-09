function [FlexNetwork,nPrune] = prunenode(FlexNetwork,winninglayer,Ey,k)
hiddenNodeSignificance = Ey;
if FlexNetwork.incrementalClass == 0
    mean_HS = mean(hiddenNodeSignificance);
    std_HS = std(hiddenNodeSignificance);
    pruneList = find(hiddenNodeSignificance < (mean_HS - 0.5*std_HS));      % Equation (7)
    if isempty(pruneList) == 1
        [~,pruneList] = min(hiddenNodeSignificance);
    end
    nPrune = numel(pruneList);
elseif FlexNetwork.incrementalClass == 1
    nPrune = FlexNetwork.nPrune;
    nWinnerNode = FlexNetwork.nNodeWinner;
    noOfPrevNode = size(Ey,2) - nWinnerNode;
    pruneCandidate = Ey(1,noOfPrevNode+1:end);
    [~,pruneIndex] = sort(pruneCandidate(1,:),2);
    pruneList = pruneIndex(1:nPrune) + noOfPrevNode;
end

FlexNetwork.W{winninglayer}(pruneList,:)       = [];
FlexNetwork.vW{winninglayer}(pruneList,:)      = [];
FlexNetwork.dW{winninglayer}(pruneList,:)      = [];

FlexNetwork.WPre{winninglayer}(pruneList,:)    = [];
FlexNetwork.deltaLossW{winninglayer}(pruneList,:) = [];
FlexNetwork.sumDeltaLossW{winninglayer}(pruneList,:) = [];
FlexNetwork.deltaW{winninglayer}(pruneList,:) = [];
FlexNetwork.weightImportanceW{winninglayer}(pruneList,:) = [];

FlexNetwork.Ws{winninglayer}(:,pruneList+1)    = [];
FlexNetwork.vWs{winninglayer}(:,pruneList+1)   = [];
FlexNetwork.dWs{winninglayer}(:,pruneList+1)   = [];

FlexNetwork.WsPre{winninglayer}(:,pruneList+1) = [];
FlexNetwork.deltaLossWs{winninglayer}(:,pruneList+1) = [];
FlexNetwork.sumDeltaLossWs{winninglayer}(:,pruneList+1) = [];
FlexNetwork.deltaWs{winninglayer}(:,pruneList+1) = [];
FlexNetwork.weightImportanceWs{winninglayer}(:,pruneList+1) = [];

if winninglayer < FlexNetwork.nHiddenLayer
    FlexNetwork.W{winninglayer+1}(:,pruneList+1)    = [];
    FlexNetwork.vW{winninglayer+1}(:,pruneList+1)   = [];
    FlexNetwork.dW{winninglayer+1}(:,pruneList+1)   = [];
    FlexNetwork.c{winninglayer+1}(pruneList,:)      = [];
    
    FlexNetwork.WPre{winninglayer+1}(:,pruneList+1) = [];
    FlexNetwork.deltaLossW{winninglayer+1}(:,pruneList+1) = [];
    FlexNetwork.sumDeltaLossW{winninglayer+1}(:,pruneList+1) = [];
    FlexNetwork.deltaW{winninglayer+1}(:,pruneList+1) = [];
    FlexNetwork.weightImportanceW{winninglayer+1}(:,pruneList+1) = [];
end
FlexNetwork.size(winninglayer+1) = FlexNetwork.size(winninglayer+1)...
    - nPrune;
% fprintf(' %d nodes are PRUNED around sample %d\n', nPrune, k)
end