function twm = totalWeightMovement(parametersInit,parameters)
% denumerator of eqn 21
twm = (parametersInit - parameters).^2;
twm = twm + 0.001*ones(size(twm));
end