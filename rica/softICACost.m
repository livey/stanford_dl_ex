%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
epsilong=1e-2;
y=W*x;
cost= params.lambda*sum(sum(sqrt(y.^2+epsilong)))+...
      .5*norm(W'*W*x-x,'fro')^2;

gd1=(y./sqrt(y.^2+epsilong))*x';
gd2=2*W*(W'*y-x)*x'+2*y*(W'*y-x)';
Wgrad =params.lambda*gd1+.5*gd2;  
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
