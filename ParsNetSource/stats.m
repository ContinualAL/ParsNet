%% Performance measure
% This function is developed from Gregory Ditzler
% https://github.com/gditzler/IncrementalLearning/blob/master/src/stats.m
function [fMeasure,gMean,recall,precision,error] = stats(trueClass, rawOutput, nClass)
label           = index2vector(trueClass, nClass);
predictedLabel  = index2vector(rawOutput, nClass);

recall      = calculate_recall(label, predictedLabel, nClass);
error       = 1 - sum(diag(predictedLabel'*label))/sum(sum(predictedLabel'*label));
precision   = calculate_precision(label, predictedLabel, nClass);
gMean       = calculate_g_mean(recall, nClass);
fMeasure    = calculate_f_measure(label, predictedLabel, nClass);


    function gMean = calculate_g_mean(recall, nClass)
        gMean = (prod(recall))^(1/nClass);
    end

    function fMeasure = calculate_f_measure(label, predictedLabel, nClass)
        fMeasure = zeros(1, nClass);
        for iClass = 1:nClass
            fMeasure(iClass) = 2*label(:, iClass)'*predictedLabel(:, iClass)/(sum(predictedLabel(:, iClass)) + sum(label(:, iClass)));
        end
        fMeasure(isnan(fMeasure)) = 1;
    end

    function precision = calculate_precision(label, predictedLabel, nClass)
        precision = zeros(1, nClass);
        for iClass = 1:nClass
            precision(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(predictedLabel(:, iClass));
        end
        precision(isnan(precision)) = 1;
    end

    function recall = calculate_recall(label, predictedLabel, nClass)
        recall = zeros(1, nClass);
        for iClass = 1:nClass
            recall(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(label(:, iClass));
        end
        recall(isnan(recall)) = 1;
    end

    function output = index2vector(input, nClass)
        output = zeros(numel(input), nClass);
        for iData = 1:numel(input)
            output(iData, input(iData)) = 1;
        end
    end
end