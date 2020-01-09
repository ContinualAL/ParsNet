function gmm = gmmUpdateWeight(gmm,Vj,expn,~)
gmm.accu = gmm.accu + expn;               % sum of activity eqn 17  
gmm.counter_accu = gmm.counter_accu + 1;  % lifespan eqn 17
denumerator = (2*pi)^(-1/2)*Vj.^(-1/2);
pxn = denumerator.*expn;                  % p(x|n)
pn  = gmm.Np./sum(gmm.Np);                % p(n)
pxnpn = pxn.*pn;                          % p(x|n)p(n)
if sum(pxnpn) == 0
    % for stability purposes
    [~,max_expj_index]    = max(expn);
    pxnpn(max_expj_index) = pxnpn(max_expj_index) + 1;
end
gmm.weight = pxnpn/sum(pxnpn);            % omega eqn 16
end