% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

%% train the winning layer
function [ParsNetwork,parameter]  = ParsNet_discriminative(ParsNetwork,...
    gmm,parameter,inputData,targetData,pseudoLabelIdentity,trueLabel)
grow  = 0;
prune = 0;
mode = ParsNetwork.mode;

%% initiate performance matrix
noOfMissLabeling = 0;
winner      = parameter.winninglayer;
kp          = parameter.ev{1}.kp;
kl          = parameter.ev{winner}.kl;
class_count = ParsNetwork.class_count(winner,:);
class_prob  = ParsNetwork.class_prob(winner,:);
K           = parameter.ev{winner}.K;
Kd          = parameter.ev{winner}.Kd;
node        = parameter.ev{winner}.node;
BIAS2       = parameter.ev{winner}.BIAS2;
VAR         = parameter.ev{winner}.VAR;
miu_NS_old  = parameter.ev{winner}.miu_NS_old;
std_NS_old  = parameter.ev{winner}.std_NS_old;
miu_NHS_old = parameter.ev{winner}.miu_NHS_old;
std_NHS_old = parameter.ev{winner}.std_NHS_old;
miumin_NS   = parameter.ev{winner}.miumin_NS;
miumin_NHS  = parameter.ev{winner}.miumin_NHS;
stdmin_NS   = parameter.ev{winner}.stdmin_NS;
stdmin_NHS  = parameter.ev{winner}.stdmin_NHS;

%% load the data for training
[nData,nin] = size(inputData);
indices     = randperm(nData);
inputData   = inputData(indices,:);
targetData  = targetData(indices,:);
if isempty(trueLabel) == 0
    nTrue = size(trueLabel,1);
    trueLabel = [zeros(nData-nTrue,ParsNetwork.nClass);trueLabel];
    trueLabel = trueLabel(indices,:);
end
pseudoLabelIdentity = pseudoLabelIdentity(indices);
ParsNetwork.WPre = ParsNetwork.W;
ParsNetwork.WsPre = ParsNetwork.Ws;
prevPseudoLabel = 0;
LpseudoLabel = [];
LAugmentedData = [];
L = [];
biasd = [];
vard = [];

%% main loop, train the model
iLabeledData = 0;
iUnlabeledData = 0;
iAugmentedData = 0;
countSI = 0;
for iData = 1 : nData
    %% calculate network significance
    if pseudoLabelIdentity(iData) == 0
        kp = kp + 1;            % counter for the whole network
        kl = kl + 1;            % counter for each hidden layer
        iLabeledData = iLabeledData + 1;
        
        %% Network mean calculation
        [~,~,Ey,~,Ez,Ez2] = netfeedforwardf(ParsNetwork,gmm,winner);
        bias2 = (Ez - targetData(iData,:)).^2;
        ns = bias2;
        NS = norm(ns,'fro');
        
        %% Incremental calculation of NS mean and variance
        [miu_NS,std_NS] = recursiveMeanStd(miu_NS_old,std_NS_old,NS,kl);
        miu_NS_old = miu_NS;
        std_NS_old = std_NS;
        miustd_NS  = miu_NS + std_NS;
        if kl <= 1 || grow == 1
            miumin_NS = miu_NS;
            stdmin_NS = std_NS;
        else
            if miu_NS < miumin_NS
                miumin_NS = miu_NS;
            end
            if std_NS < stdmin_NS
                stdmin_NS = std_NS;
            end
        end
        miustdmin_NS  = miumin_NS + (1.2*exp(-NS)+0.8)*stdmin_NS;
        BIAS2(kl,:)   = miu_NS;
        biasd(iData) = miu_NS;
        
        %% growing hidden unit
        if miustd_NS >= miustdmin_NS && kl > 1 && mode(2) == 1 &&...
                prevPseudoLabel == 0
            grow  = 1;
            ParsNetwork = grownode(ParsNetwork,gmm.k,winner,kp);
            
            K = K  + gmm.k;
            Kd = Kd + gmm.k;
            node(kl)  = K;
            noded(kl) = Kd;
        else
            grow     = 0;
            node(kl) = K;
            noded(kl)= Kd;
        end
        
        %% Network variance calculation
        var = Ez2 - Ez.^2;
        NHS = norm(var,'fro');
        
        %% Incremental calculation of NHS mean and variance
        [miu_NHS,std_NHS] = recursiveMeanStd(miu_NHS_old,...
            std_NHS_old,NHS,kl);
        miu_NHS_old = miu_NHS;
        std_NHS_old = std_NHS;
        miustd_NHS  = miu_NHS + std_NHS;
        if kl <= nin+1 || prune == 1
            miumin_NHS = miu_NHS;
            stdmin_NHS = std_NHS;
        else
            if miu_NHS < miumin_NHS
                miumin_NHS = miu_NHS;
            end
            if std_NHS < stdmin_NHS
                stdmin_NHS = std_NHS;
            end
        end
        miustdmin_NHS = miumin_NHS + 2*(1.2*exp(-NHS)+0.8)*stdmin_NHS;
        VAR(kl,:) = miu_NHS;
        vard(iData) = miu_NHS;
        
        %% pruning hidden unit
        if grow == 0 && Kd > 0 && miustd_NHS >= miustdmin_NHS &&...
                kl > nin + 1 && mode(3) == 1 && prevPseudoLabel == 0
            prune = 1;
            [ParsNetwork,nPrune] = prunenode(ParsNetwork,winner,Ey,kp);
            
            K = K - nPrune;
            Kd = Kd - nPrune;
            node(kl) = K;
            noded(kl) = Kd;
        else
            node(kl) = K;
            noded(kl)= Kd;
            prune = 0;
        end
        
        %% feedforward
        x_tail = maskingnoise(inputData(iData,:),nin,0);    % 0: without noise
        ParsNetwork.a = {};
        ParsNetwork.as = {};
        ParsNetwork.e = {};
        ParsNetwork = netFeedForward(ParsNetwork, x_tail, targetData(iData,:));
        L(iLabeledData,:) = ParsNetwork.L*ParsNetwork.betaOld';%*(1 - parameter.lambda);
        
        %% update with labeled data
        ParsNetwork = netbackpropagation(ParsNetwork,winner);
        if prevPseudoLabel == 0
            originalLabeledSamples = 1;
            ParsNetwork = netupdate(ParsNetwork,winner,originalLabeledSamples);
        elseif prevPseudoLabel == 1
            % calculate regularizer strength alpha3
            if parameter.t > 10
                [~,~,~,L_gen] = feedforwarddae(...
                    ParsNetwork,x_tail,x_tail,winner);  % feedforward DAE
                ParsNetwork.regStr = calcRegStr(L_gen,parameter.dae{1}.maxError,...
                    parameter.dae{1}.minError);     % calculate regularizer strength alpha3
            end
            
            % initiate previous weight to track the total weight movement
            ParsNetwork.WPre  = ParsNetwork.W;
            ParsNetwork.WsPre = ParsNetwork.Ws;
            trueLabels = 1;
            ParsNetwork = netupdate_unlabeled(ParsNetwork,winner,Wstar,WsStar,trueLabels);
        end
        
        prevPseudoLabel = 0;
    elseif pseudoLabelIdentity(iData) == 1 && mode(6) == 1
        iAugmentedData = iAugmentedData + 1;
        
        %% feedforward
        x_tail = maskingnoise(inputData(iData,:),nin,0);
        ParsNetwork.a = {};
        ParsNetwork.as = {};
        ParsNetwork.e = {};
        ParsNetwork = netFeedForward(ParsNetwork, x_tail, targetData(iData,:));
        LAugmentedData(iAugmentedData,:) = ParsNetwork.L*ParsNetwork.betaOld';
        
        %% update with augmented data
        ParsNetwork = netbackpropagation(ParsNetwork,winner);
        if prevPseudoLabel == 0
            originalLabeledSamples = 1;
            ParsNetwork = netupdate(ParsNetwork,winner,originalLabeledSamples);
        elseif prevPseudoLabel == 1
            % calculate regularizer strength alpha3
            if parameter.t > 10
                [~,~,~,L_gen] = feedforwarddae(...
                    ParsNetwork,x_tail,x_tail,winner);  % feedforward DAE
                ParsNetwork.regStr = calcRegStr(L_gen,parameter.dae{1}.maxError,...
                    parameter.dae{1}.minError);     % calculate regularizer strength alpha3
            end
            
            % initiate previous weight
            ParsNetwork.WPre  = ParsNetwork.W;
            ParsNetwork.WsPre = ParsNetwork.Ws;
            trueLabels = 1;
            ParsNetwork = netupdate_unlabeled(ParsNetwork,winner,Wstar,WsStar,trueLabels);
        end
        
        prevPseudoLabel = 0;
    elseif pseudoLabelIdentity(iData) == 2 && mode(6) == 1
        iUnlabeledData  = iUnlabeledData + 1;
        if prevPseudoLabel == 0
            %% weight importance
            Wstar = ParsNetwork.W;
            WsStar = ParsNetwork.Ws;
            ParsNetwork = weightImportance(ParsNetwork,parameter.winninglayer,parameter.t);
        else
            countSI = countSI + 1;
        end
        prevPseudoLabel = 1;
        
        %% feedforward
        x_tail = maskingnoise(inputData(iData,:),nin,0);
        ParsNetwork.a  = {};
        ParsNetwork.as = {};
        ParsNetwork.e  = {};
        ParsNetwork    = netFeedForward(ParsNetwork, x_tail, targetData(iData,:));
        
        % calculate regularizer strength alpha3
        if parameter.t > 10
            [~,~,~,L_gen] = feedforwarddae(...
                ParsNetwork,x_tail,x_tail,winner);  % feedforward DAE
            ParsNetwork.regStr = calcRegStr(L_gen,parameter.dae{1}.maxError,...
                parameter.dae{1}.minError);         % calculate regularizer strength alpha3
        end
        
        %% update with pseudo label
        trueLabels  = 0;
        ParsNetwork = netbackpropagation(ParsNetwork,winner);
        ParsNetwork = netupdate_unlabeled(ParsNetwork,winner,Wstar,WsStar,trueLabels);
        
        %% calculate loss
        ParsNetwork = netFeedForward(ParsNetwork, x_tail, trueLabel(iData,:));
        if sum(targetData(iData,:) == trueLabel(iData,:)) ~= ParsNetwork.nClass
            noOfMissLabeling = noOfMissLabeling + 1;
        end
        LpseudoLabel(iUnlabeledData,:) = ParsNetwork.L*ParsNetwork.betaOld';
    end
end
if prevPseudoLabel == 0
    ParsNetwork = weightImportance(ParsNetwork,parameter.winninglayer,parameter.t);
end

if isempty(LpseudoLabel) == 1
    LpseudoLabel = 0;
end
if isempty(L) == 1
    L = 0;
end
if isempty(LAugmentedData) == 1
    LAugmentedData = 0;
end


%% reset momentum and gradient
ParsNetwork = resetGradient(ParsNetwork,0,0);

%% substitute the recursive calculation
parameter.ev{1}.noOfMissLabeling = noOfMissLabeling;
parameter.ev{1}.kp               = kp;
parameter.ev{winner}.kl          = kl;
ParsNetwork.class_count(winner,:)= class_count;
ParsNetwork.class_prob(winner,:) = class_prob;
parameter.ev{winner}.K           = K;
parameter.ev{winner}.Kd          = Kd;
parameter.ev{winner}.node        = node;
parameter.ev{winner}.LF          = mean(L);
parameter.ev{winner}.LpseudoLabel= mean(LpseudoLabel);
parameter.ev{winner}.LAugmentedData= mean(LAugmentedData);
parameter.ev{winner}.BIAS2       = BIAS2;
parameter.ev{winner}.VAR         = VAR;
parameter.ev{winner}.meanBIAS2   = mean(biasd);
parameter.ev{winner}.meanVAR     = mean(vard);
parameter.ev{winner}.miu_NS_old  = miu_NS_old;
parameter.ev{winner}.std_NS_old  = std_NS_old;
parameter.ev{winner}.miu_NHS_old = miu_NHS_old;
parameter.ev{winner}.std_NHS_old = std_NHS_old;
parameter.ev{winner}.miumin_NS   = miumin_NS;
parameter.ev{winner}.miumin_NHS  = miumin_NHS;
parameter.ev{winner}.stdmin_NS   = stdmin_NS;
parameter.ev{winner}.stdmin_NHS  = stdmin_NHS;
end