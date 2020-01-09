function [hiddenRepresentation,x_hat,err_gen,L_gen] = feedforwarddae(...
    FlexNetwork,x_tail,x_tail_ori,winninglayer)
a = FlexNetwork.W{winninglayer}(:,2:end)*x_tail' + ...
    FlexNetwork.W{winninglayer}(:,1);
hiddenRepresentation = sigmf(a,[1,0]);
a_hat = FlexNetwork.W{winninglayer}(:,2:end)'*hiddenRepresentation +...
    FlexNetwork.c{winninglayer};
x_hat = sigmf(a_hat,[1,0]);
x_hat = x_hat';
err_gen = x_tail_ori - x_hat;
L_gen = 0.5*sum(err_gen)^2;
end