% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

%% main code
% for sporadic access to ground truth scenario
function [ParsNetwork,Agmm,parameter,performance] = ParsNet(data,nin,mode,dType,dataProportion,chunkSize)
%% divide the data into nFolds chunks
addpath('AGMM')
fprintf('=========ParsNet is started=========\n')
stream = RandStream('mt19937ar','Seed',0);
[Data,chunk_size,nClass,nFolds] = dataprep(data,nin,chunkSize);
tTest = []; act = []; out = [];
nAdditionalSample = [];
clear data

%% initiate model
[ParsNetwork,parameter] = ParsNet_init(nin,nClass);
ParsNetwork.mode = mode;

%% initiate gmm calculation
Agmm = gmmInitialize(nin,nClass,Data{1}(1,1:nin),mode);

%% initiate Evaluation Window
parameter.ev{1}.Q  = chunk_size;        %initial sliding window size
parameter.dae{1}.Q = chunk_size;

%% main loop, prequential evaluation
for iFolds = 1:nFolds
    %% load the data chunk-by-chunk
    inputData  = Data{iFolds}(:,1:nin);
    targetData = Data{iFolds}(:,nin+1:end);
    [nData,~]  = size(targetData);
    clear Data{t}
    
    %% neural network testing
    parameter.t = iFolds;
    
    %% obtain labeled data
    [inputLabeledData,targetLabeledData,nLabeledData,...
        inputUnlabeledData,targetUnlabeledData,~] = ...
        reduceShuffleData(inputData,targetData,nData,dataProportion,stream);
    inputData = [inputLabeledData;inputUnlabeledData];
    targetData = [targetLabeledData;targetUnlabeledData];
    parameter.x = inputData;
    parameter.T = targetData;
    parameter.inputLabeledData = inputLabeledData;
    parameter.targetLabeledData = targetLabeledData;
    noOfLabeledData = size(inputLabeledData,1);
    nLabeledSamples(iFolds) = noOfLabeledData;
    noOfUnlabeledData = size(inputUnlabeledData,1);
    
    %% testing phase
    start_test = tic;
    [ParsNetwork,parameter] = netTest(ParsNetwork,parameter,...
        inputData,targetData,nLabeledData);
    unlabeledMultiClassProb = ParsNetwork.sigma(nLabeledData+1:end,:);
    parameter.test_time(iFolds) = toc(start_test);
    
    %% metrics calculation
    parameter.Loss(iFolds) = ParsNetwork.L(parameter.winninglayer);
    tTest(nData*iFolds+(1-nData):nData*iFolds,:) = ParsNetwork.sigma;
    if iFolds > 1
        act = [act parameter.act'];
        out = [out ParsNetwork.out'];
    end
    parameter.acc_residual_error...
        (nData*iFolds+(1-nData):nData*iFolds,:) = parameter.residual_error;
    parameter.acc_cr(iFolds)                = parameter.cr;
    parameter.MeanCr(iFolds)                = mean(parameter.acc_cr);
    
    %% statistical measure
    [performance.ev.f_measure(iFolds,:),performance.ev.g_mean(iFolds,:),...
        performance.ev.recall(iFolds,:),performance.ev.precision(iFolds,:),...
        performance.ev.err(iFolds,:)] = stats(parameter.act, ...
        ParsNetwork.out, nClass);
    if iFolds == nFolds
        fprintf(...
            '=========ParsNet is finished=========\n')
        break               % last chunk only testing
    end
    
    start_train = tic;
    if iFolds > 1
        %% calculate evaluation window for discriminative and generative
        ParsNetwork.grow_layer = 0;
        parameter.ev{1}.grow_layer  = 0;
        parameter.dae{1}.grow_layer = 0;
    end
    
    %% data augmentation
    if mode(6) == 1
        switch dType
            case 'image'
                [augmentedInputLabeledData,augmentedTargetLabeledData] =...
                    addNoise(inputLabeledData,targetLabeledData,2,0.13);
            case {'sensor'}
                [augmentedInputLabeledData,augmentedTargetLabeledData] =...
                    addNoise(inputLabeledData,targetLabeledData,2,0.001);
        end
    elseif mode(6) == 0
        augmentedInputLabeledData = inputLabeledData;
        augmentedTargetLabeledData = targetLabeledData;
    end
    augmentedLabelIdentity = [zeros(1,size(inputLabeledData,1))...
        ones(1,size(augmentedInputLabeledData,1) - size(inputLabeledData,1))];
    
    %% Self labeling
    if iFolds > 10 && dataProportion ~= 1 && mode(6) == 1
        [additionalInputLabeledData,additionalTargetLabeledData,trueLabel] = ...
            selfLabelling(Agmm,inputUnlabeledData,...
            unlabeledMultiClassProb,targetUnlabeledData);
        finalInput = [augmentedInputLabeledData;additionalInputLabeledData];
        finalTarget = [augmentedTargetLabeledData;additionalTargetLabeledData];
        pseudoLabelIdentity = [augmentedLabelIdentity 2*ones(1,...
            size(additionalInputLabeledData,1))];
        noOfPseudolabeledData = size(additionalInputLabeledData,1);
        nUnlabeledSamples(iFolds) = noOfUnlabeledData;
    elseif dataProportion == 1 || iFolds <= 10 || mode(6) == 0
        additionalTargetLabeledData = [];
        finalInput = augmentedInputLabeledData;
        finalTarget = augmentedTargetLabeledData;
        pseudoLabelIdentity = augmentedLabelIdentity;
        trueLabel = [];
        noOfPseudolabeledData = 0;
        nUnlabeledSamples(iFolds) = 0;
    end
    nAdditionalSample(iFolds) = size(additionalTargetLabeledData,1);
    
    %% Generative training
    InputDataGenerative = ParsNetwork.a{parameter.winninglayer};
    Agmm.calculation = 1;      % Turn on the AGMM
    [ParsNetwork,Agmm,parameter] = ParsNet_generative(ParsNetwork,...
        Agmm,parameter,InputDataGenerative,inputData,...
        targetLabeledData,nLabeledData);
    ParsNetwork.sizeGenerative = ParsNetwork.size;
    parameter.dae{parameter.winninglayer}.nodeGen(iFolds) = parameter.dae{parameter.winninglayer}.K;
    parameter.dae{1}.Loss(iFolds) = ...
        parameter.dae{parameter.winninglayer}.LF;
    parameter.dae{1}.bias(iFolds) = parameter.dae{parameter.winninglayer}.meanBIAS2;
    parameter.dae{1}.var(iFolds) = parameter.dae{parameter.winninglayer}.meanVAR;
    
    %% Discrinimanive training 
    [ParsNetwork,parameter] = ParsNet_discriminative(ParsNetwork,...
        Agmm,parameter,finalInput,finalTarget,pseudoLabelIdentity,trueLabel);
    noOfMissLabeling(iFolds) = parameter.ev{1}.noOfMissLabeling;
    parameter.ev{parameter.winninglayer}.nodeDisc(iFolds) = parameter.ev{parameter.winninglayer}.K;
    parameter.ev{1}.Loss(iFolds) = ...
        parameter.ev{parameter.winninglayer}.LF;
    parameter.ev{1}.LossPseudoLabel(iFolds) = ...
        parameter.ev{parameter.winninglayer}.LpseudoLabel;
    parameter.ev{1}.LAugmentedData(iFolds) = ...
        parameter.ev{parameter.winninglayer}.LAugmentedData;
    parameter.ev{1}.bias(iFolds) = parameter.ev{parameter.winninglayer}.meanBIAS2;
    parameter.ev{1}.var(iFolds) = parameter.ev{parameter.winninglayer}.meanVAR;
    parameter.update_time(iFolds) = toc(start_train);
    parameter.totalLoss(iFolds) = parameter.dae{1}.Loss(iFolds) + ...
        parameter.ev{1}.Loss(iFolds) + parameter.ev{1}.LossPseudoLabel(iFolds) +...
        parameter.ev{1}.LAugmentedData(iFolds);
    
    %% clear current chunk data
    clear Data{t}
    ParsNetwork.a = {};
    
    %% print result
    if mod(iFolds,10) == 0 || iFolds == 2
        fprintf('Minibatch: %d/%d\n', iFolds, nFolds);
        fprintf('Max Mean Min Now CR: %f%% %f%% %f%% %f%%\n', max(parameter.acc_cr(2:end))*100, mean(parameter.acc_cr(2:end))*100, min(parameter.acc_cr(2:end))*100, parameter.acc_cr(iFolds)*100);
        fprintf('Max Mean Min Now Loss: %f %f %f %f\n', max(parameter.totalLoss(2:end)), mean(parameter.totalLoss(2:end)), min(parameter.totalLoss(2:end)), parameter.totalLoss(iFolds));
        fprintf('Max Mean Min Now Accu Training time: %f %f %f %f %f\n', max(parameter.update_time), mean(parameter.update_time), min(parameter.update_time), parameter.update_time(iFolds), sum(parameter.update_time));
        fprintf('Max Mean Min Now Accu Testing time: %f %f %f %f %f\n', max(parameter.test_time), mean(parameter.test_time), min(parameter.test_time), parameter.test_time(iFolds), sum(parameter.test_time));
        fprintf('Network structure: %s (Discriminative) | %s (Generative)\n', num2str(ParsNetwork.size(:).'), num2str(ParsNetwork.sizeGenerative(:).'));
        fprintf('Total of samples: %d Labeled | %d Unlabeled | %d Selflabeled\n', noOfLabeledData, noOfUnlabeledData, noOfPseudolabeledData);
        fprintf('\n');
    end
    parameter.hiddenNodeGenEvo(iFolds) = ParsNetwork.sizeGenerative(2);
    parameter.hiddenNodeDisEvo(iFolds) = ParsNetwork.size(2);
    Agmm.kEvo(iFolds) = Agmm.k;
end

%% statistical measure
[performance.f_measure,performance.g_mean,performance.recall,...
    performance.precision,performance.err] = stats(act, out, nClass);

%% save the numerical result
parameter.nFolds = nFolds;
performance.update_time = [mean(parameter.update_time)...
    std(parameter.update_time)];
performance.test_time = [mean(parameter.test_time)...
    std(parameter.test_time)];
performance.classification_rate = [mean(parameter.acc_cr(2:end))...
    std(parameter.acc_cr(2:end))];
performance.totalLoss = [mean(parameter.totalLoss(2:end))...
    std(parameter.totalLoss(2:end))];
performance.nAdditionalSample = [mean(nAdditionalSample),...
    std(nAdditionalSample)];
performance.nMisslabeledSample = [mean(noOfMissLabeling),...
    std(noOfMissLabeling)];
performance.LayerWeight = ParsNetwork.beta;
meanode = [];
stdnode = [];
for iHiddenLayer = 1:ParsNetwork.nHiddenLayer
    a = nnz(~ParsNetwork.nodes{iHiddenLayer});
    ParsNetwork.nodes{iHiddenLayer} = ParsNetwork.nodes{iHiddenLayer}...
        (a+1:iFolds);
    meanode = [meanode mean(ParsNetwork.nodes{iHiddenLayer})];
    stdnode = [stdnode std(ParsNetwork.nodes{iHiddenLayer})];
end
performance.meanode = meanode;
performance.stdnode = stdnode;
performance.NumberOfParameters = parameter.mnop;

%% print result
disp('Final Result')
fprintf('Mean Std Classification Rate: %f %f\n', mean(parameter.acc_cr(2:end))*100, std(parameter.acc_cr(2:end))*100);
fprintf('Mean Std training time: %f %f\n', mean(parameter.update_time), std(parameter.update_time));
fprintf('Mean Std testing time: %f %f\n', mean(parameter.test_time), std(parameter.test_time));
fprintf('Mean Std Number of hidden nodes: %f %f\n', meanode, stdnode);
fprintf('Mean Std Number of additional samples: %f %f\n', mean(nAdditionalSample), std(nAdditionalSample));
fprintf('Mean Std Number of parameters: %f %f\n', parameter.mnop(1),parameter.mnop(2));
fprintf('\n\n');
end
