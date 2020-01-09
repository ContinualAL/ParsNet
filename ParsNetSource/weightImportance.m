function net = weightImportance(net,winninglayer,~)
net.weightImportanceCount = net.weightImportanceCount + 1;
net.WPost = net.W;
net.WsPost = net.Ws;

%% normalize the weight importance
% To avoid exploding gradien, here we apply weight normalization 
% whenever the max of weight importance is greater than 500
if max(max(net.weightImportanceW{winninglayer})) > 500
    net.weightImportanceW{winninglayer} = net.weightImportanceW{winninglayer}/norm(net.weightImportanceW{winninglayer});
end
if max(max(net.weightImportanceWs{winninglayer})) > 500
    net.weightImportanceWs{winninglayer} = net.weightImportanceWs{winninglayer}/norm(net.weightImportanceWs{winninglayer});
end

%% calculate normalized weight movement
net.W_twm{winninglayer} = totalWeightMovement(net.WPre{winninglayer},...
    net.WPost{winninglayer});       % denumerator of eqn 21 for input weight
net.Ws_twm{winninglayer} = totalWeightMovement(net.WsPre{winninglayer},...
    net.WsPost{winninglayer});      % denumerator of eqn 21 for output weight

%% calculate weight importance
weightImportanceW = net.sumDeltaLossW{winninglayer}./...
    net.W_twm{winninglayer};        % eqn 21 for input weight
weightImportanceWs = net.sumDeltaLossWs{winninglayer}./...
    net.Ws_twm{winninglayer};       % eqn 21 for output weight

weightImportanceW  = abs(weightImportanceW);
weightImportanceWs = abs(weightImportanceWs);

%% accumulate the weight importance
net.weightImportanceW{winninglayer} = (net.weightImportanceW{winninglayer}+...
    weightImportanceW);
net.weightImportanceWs{winninglayer} = (net.weightImportanceWs{winninglayer}+...
    weightImportanceWs);

%% reset
net.sumDeltaLossW{winninglayer} = net.sumDeltaLossW{winninglayer}*0;
net.sumDeltaLossWs{winninglayer}= net.sumDeltaLossWs{winninglayer}*0;
net.WPre = {};
net.W_twm = {};
net.WsPre = {};
net.WPost = {};
net.WsPost = {};
net.Ws_twm = {};
end