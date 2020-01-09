% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

function [net,parameter] = netTest(net, parameter, input,trueClass,...
    nLabeledData)
%% feedforward
net = netFeedForward(net, input, trueClass);
[nData,nClass] = size(trueClass);

%% obtain trueclass label
[~,acttualLabel] = max(trueClass,[],2);

net.sigma = zeros(nData,nClass);
countCorrectPrediction = zeros(1,net.nHiddenLayer);
for iData = 1 : nData
    for iHiddenLayer = 1 : net.nHiddenLayer
        % this is designed for multilayer network. In the case of FlexNet,
        % no voting mechanism is applied. net.betaOld is always 1.
        if net.beta(iHiddenLayer) ~= 0
            %% obtain the predicted label
            % note that the layer weight betaOld is fixed
            net.sigma(iData,:) = net.sigma(iData,:) + ...
                net.as{iHiddenLayer}(iData,:)*net.betaOld(iHiddenLayer);
            [~, net.classlabel{iHiddenLayer}(iData,:)] =...
                max(net.as{iHiddenLayer}(iData,:),[],2);
            compare = acttualLabel(iData,:) == net.classlabel{iHiddenLayer}(iData,:);
            
            %% train the weighted voting
            if compare == 0 && iData <= nLabeledData
                net.beta(iHiddenLayer) = max(net.beta(iHiddenLayer)*...
                    net.penalty(iHiddenLayer),1/net.nHiddenLayer);
                net.penalty(iHiddenLayer) = max(net.penalty(iHiddenLayer)...
                    -0.001,0.001);
            elseif compare == 1 && iData <= nLabeledData
                net.beta(iHiddenLayer) = min(net.beta(iHiddenLayer)*...
                    (1+net.penalty(iHiddenLayer)),1);
                net.penalty(iHiddenLayer) = min(net.penalty(iHiddenLayer)...
                    +0.001,1);
                countCorrectPrediction(iHiddenLayer) = 1+...
                    countCorrectPrediction(iHiddenLayer);
            end
        end
        
        if iData == nData
            %% calculate the number of parameter
            if net.beta(iHiddenLayer) ~= 0
                [c,d] = size(net.Ws{iHiddenLayer});
                vw = 1;
            else
                c = 0;
                d = 0;
                vw = 0;
            end
            [a,b] = size(net.W{iHiddenLayer});
            nParameters(iHiddenLayer) = a*b + c*d + vw;
            
            %% calculate the number of node in each hidden layer
            net.nodes{iHiddenLayer}(parameter.t) = ...
                size(net.W{iHiddenLayer},1);
        end
    end
end
parameter.nop(parameter.t) = sum(nParameters);
parameter.mnop = [mean(parameter.nop) std(parameter.nop)];

%% update the voting weight
net.beta  = net.beta./sum(net.beta);
net.betaOld = net.beta;
[~,parameter.winninglayer] = max(net.beta);

%% calculate classification rate
[raw_out,predictedLabel] = max(net.sigma,[],2);
parameter.bad            = find(predictedLabel ~= acttualLabel);
parameter.cr             = 1 - numel(parameter.bad)/nData;
parameter.residual_error = 1 - raw_out;
net.out                  = predictedLabel;
parameter.act            = acttualLabel;
parameter.badFromLabeled = find(predictedLabel(1:nLabeledData,:) ~=...
    acttualLabel(1:nLabeledData,:));
end