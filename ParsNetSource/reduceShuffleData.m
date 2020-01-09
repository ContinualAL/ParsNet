function [inputLabeledData,targetLabeledData,nLabeledData,...
    inputUnlabeledData,targetUnlabeledData,nUnlabeledData] = ...
    reduceShuffleData(inputData,targetData,nData,dataProportion,stream)
ApplyPermutation = randperm(stream,nData);
inputData  = inputData(ApplyPermutation,:);
targetData = targetData(ApplyPermutation,:);
if dataProportion ~= 1
    nLabeledData = round(dataProportion*nData);
    inputLabeledData  = inputData(1:nLabeledData,:);
    targetLabeledData = targetData(1:nLabeledData,:);
    nUnlabeledData = nData - nLabeledData;
    inputUnlabeledData = inputData(nLabeledData+1:end,:);
    targetUnlabeledData = targetData(nLabeledData+1:end,:);
elseif dataProportion == 1
    nLabeledData = nData;
    inputLabeledData = inputData;
    targetLabeledData = targetData;
    nUnlabeledData = 0;
    inputUnlabeledData = [];
    targetUnlabeledData = [];
end
end