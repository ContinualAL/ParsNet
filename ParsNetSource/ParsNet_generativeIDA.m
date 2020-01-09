% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

%% evolving denoising autoencoder
% for infinite delay scenario
function [ParsNetwork,gmm,parameter] = ParsNet_generativeIDA(ParsNetwork,...
    gmm,parameter,x,x_ori,targetLabeledData)
winner = parameter.winninglayer;

%% initiate parameter
if gmm.calculation == 1
    [nx_ori,nin] = size(x_ori);         % input for the AGMM
end
x_tail             = x(:,2:end);        % input for the current layer
[nData,nInput]     = size(x_tail);
x_tail_ori         = x_tail;
W                  = ParsNetwork.W{winner}(:,2:end);
c                  = ParsNetwork.c{winner};
[nNode,~]          = size(W);

%% a parameter to indicate if there is growing/pruning
grow = 0;
prune = 0;
mode = ParsNetwork.mode;
evaluationWindow = nData;

%% initiate performance matrix
miu_NS_old     = parameter.dae{winner}.miu_NS_old;
std_NS_old     = parameter.dae{winner}.std_NS_old;
miu_NHS_old    = parameter.dae{winner}.miu_NHS_old;
std_NHS_old    = parameter.dae{winner}.std_NHS_old;
miumin_NS      = parameter.dae{winner}.miumin_NS;
miumin_NHS     = parameter.dae{winner}.miumin_NHS;
stdmin_NS      = parameter.dae{winner}.stdmin_NS;
stdmin_NHS     = parameter.dae{winner}.stdmin_NHS;
nodeg          = parameter.dae{winner}.node;
Kg             = parameter.dae{winner}.Kg;
kk             = parameter.dae{1}.kk;
ky             = parameter.dae{winner}.ky;
node           = parameter.ev{winner}.node;
BIAS2          = parameter.dae{winner}.BIAS2;
VAR            = parameter.dae{winner}.VAR;

ParsNetwork.WPre  = ParsNetwork.W;
ParsNetwork.WsPre = ParsNetwork.Ws;

%% Generative training
for iData = 1:nx_ori
    %% AGMM 
    % update agmm using new data and augmented data
    if gmm.calculation == 1
        if iData > 1 && iData <= nx_ori
            if iData <= nData
                meannetbias = meanNetBias2(iData-1);
            elseif iData > nData
                meannetbias = meanNetBias;
            end
            gmm = AGMMID(gmm,x_ori(iData,:),nin,meannetbias,...
                evaluationWindow,nData,iData,mode);
            if iData > nData % only updated when there are labels
                [gmm.Nc,gmm.Ni] = gmmUpdateLabel(gmm.gmmwinner,...
                    targetLabeledData(iData,:),gmm.Nc,gmm.Ni);
            end
        end
    end
    
    if iData <= nData       % update the network using new incoming data
        kk = kk + 1;        % counter for the whole network
        ky = ky + 1;        % counter for each hidden layer
        
        %% Input masking
        x_tail(iData,:) = maskingnoise(x_tail(iData,:),nInput,0.1);
        
        %% calculate network significance
        [EzNet,~,Ey,Ey2,~,~] = netfeedforwardf(ParsNetwork,...
            gmm,winner);
        Ez = sigmf(W'*Ey'  + c,[1,0]);
        Ez2 = sigmf(W'*Ey2' + c,[1,0]);
        
        netBias2(iData,:) = norm((EzNet - parameter.T(iData,:)).^2,'fro');
        meanNetBias2(iData) = mean(netBias2);
        if iData == nData
            meanNetBias = meanNetBias2(iData);
        end
        
        %% hidden layer growing signal
        if mode(1) == 1
            %% winning layer bias2 calculation
            bias2 = (Ez - x_tail_ori(iData,:)').^2;
            ns = bias2;
            NS = mean(ns);
            
            %% Incremental calculation of NS mean and variance
            [miu_NS,std_NS] = recursiveMeanStd(miu_NS_old,std_NS_old,NS,ky);
            BIAS2(ky,:) = miu_NS;
            biasd(iData) = miu_NS;
            miu_NS_old = miu_NS;
            std_NS_old = std_NS;
            miustd_NS = miu_NS + std_NS;
            if ky <= 1 || grow == 1
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
            miustdmin_NS = miumin_NS + (1.2*exp(-NS)+0.8)*stdmin_NS;
            
            %% growing hidden unit
            if miustd_NS >= miustdmin_NS && ky > 1 && mode(2) == 1
                % term count > add_old is added to ensure that the bias measured
                % is valid after hidden node growing
                grow = 1;
                ParsNetwork = grownode(ParsNetwork,gmm.k,winner,kk);
                nNode = nNode + gmm.k;
                Kg = Kg + gmm.k;
                node(ky) = nNode;
                nodeg(ky) = Kg;
            else
                grow = 0;
                node(ky) = nNode;
                nodeg(ky) = Kg;
            end
            
            %% Network variance calculation
            var = Ez2 - Ez.^2;
            NHS = mean(var);
            
            %% Incremental calculation of NHS mean and variance
            [miu_NHS,std_NHS] = recursiveMeanStd(miu_NHS_old,std_NHS_old,...
                NHS,ky);
            VAR(ky,:) = miu_NHS;
            vard(iData) = miu_NHS;
            miu_NHS_old = miu_NHS;
            std_NHS_old = std_NHS;
            miustd_NHS = miu_NHS + std_NHS;
            if ky <= nInput + 1 || prune == 1
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
            
            %% pruning hidden unit
            if grow == 0 && Kg > 1 && miustd_NHS >= miustdmin_NHS &&...
                    ky > nInput + 1 && mode(3) == 1
                prune = 1;
                [ParsNetwork,nPrune] = prunenode(ParsNetwork,winner,Ey,kk);
                nNode = nNode - nPrune;
                Kg = Kg - nPrune;
                node(ky) = nNode;
                nodeg(ky) = Kg;
            else
                node(ky) = nNode;
                nodeg(ky) = Kg;
                prune = 0;
            end
            
            %% feedforward
            [hiddenRepresentation,x_hat,err_gen,L_gen] = feedforwarddae(...
                ParsNetwork,x_tail(iData,:),x_tail_ori(iData,:),winner);
            L(iData,:) = L_gen;
            
            %% Backpropaagation of DAE, tied weight
            ParsNetwork = backpropdae(ParsNetwork,x_tail(iData,:),x_hat,err_gen,...
                hiddenRepresentation,winner);
            W = ParsNetwork.W{winner}(:,2:end);
            c = ParsNetwork.c{winner};
        else
            L = 0;
        end
    end
end

%% calculate regularizer strength
if parameter.t > 10
    maxL = max(L);
    minL = min(L);
    if maxL > parameter.dae{1}.maxError
        parameter.dae{1}.maxError = maxL;
    end
    if minL < parameter.dae{1}.minError
        parameter.dae{1}.minError = minL;
    end
end

%% substitute the weight back to evdae
ParsNetwork.K(winner)                = nNode;
parameter.dae{winner}.node           = nodeg;
parameter.ev{winner}.node            = node;
parameter.dae{winner}.LF             = mean(L);
parameter.dae{winner}.BIAS2          = BIAS2;
parameter.dae{winner}.VAR            = VAR;
parameter.dae{winner}.meanBIAS2      = mean(biasd);
parameter.dae{winner}.meanVAR        = mean(vard);
parameter.dae{1}.kk                  = kk;
parameter.dae{winner}.ky             = ky;
parameter.dae{winner}.K              = nNode;
parameter.dae{winner}.Kg             = Kg;
parameter.ev{winner}.K               = nNode;
parameter.dae{winner}.miu_NS_old     = miu_NS_old;
parameter.dae{winner}.std_NS_old     = std_NS_old;
parameter.dae{winner}.miu_NHS_old    = miu_NHS_old;
parameter.dae{winner}.std_NHS_old    = std_NHS_old;
parameter.dae{winner}.miumin_NS      = miumin_NS;
parameter.dae{winner}.miumin_NHS     = miumin_NHS;
parameter.dae{winner}.stdmin_NS      = stdmin_NS;
parameter.dae{winner}.stdmin_NHS     = stdmin_NHS;
end