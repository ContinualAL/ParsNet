% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

function FlexNetwork = backpropdae(FlexNetwork,x_tail,x_hat,e,h,...
    winninglayer)
lr      = FlexNetwork.learningRateDae;
W_old   = FlexNetwork.W{winninglayer}(:,2:end);
b_old   = FlexNetwork.W{winninglayer}(:,1);
c_old   = FlexNetwork.c{winninglayer};

%% calculate gradient, tied weight
dedxhat = -e;
del_j   = x_hat.*(1 - x_hat);
d3      = dedxhat.*del_j;
d_act   = (h.*(1 - h))';
d2      = (d3 * FlexNetwork.W{winninglayer}(:,2:end)') .* d_act;
dW2     = (d3' * h');
dW1     = (d2' * x_tail);
dW      = dW1 + dW2';           % weight gradient
del_W   = del_j.*W_old.*d_act';
dedb    = dedxhat*del_W';       % bias hidden layer gradient
dejdcj  = dedxhat.*del_j;       % bias output layer gradient

%% apply gradient
FlexNetwork.W{winninglayer}(:,2:end) = W_old - lr*dW;
FlexNetwork.W{winninglayer}(:,1)     = b_old - lr*dedb';
FlexNetwork.c{winninglayer}          = c_old - lr*dejdcj';
end