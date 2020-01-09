function regularizerStrength = calcRegStr(error,maxError,minError)
%% calculate alpha3, eqn 19
span = maxError - minError;
regularizerStrength = (error - minError)/span;
end

