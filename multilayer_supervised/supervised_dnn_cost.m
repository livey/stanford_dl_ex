function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%

% firstly, decide the activation function and it's corresponding
% derivatives 
numSamples = size(data, 2);
switch ei.activation_fun
    case 'logistic'
        actF = @sigmoid ;
        g_actF = @g_sigmoid;
    case 'tanh'
        actF = @tanh ;
        g_actF = @g_tanh;
    case 'relu'
        actF = @relu ; 
        g_actF = @g_relu ; 
    case 'linear'
        actF = @linea;
        g_actF = @g_linea;
end

for ll = 1:numHidden
    if ll>1 
        hAct{ll}=stack{ll}.W*hAct{ll-1} + repmat(stack{ll}.b,1,numSamples);
    else
        hAct{ll}=stack{ll}.W*data + repmat(stack{ll}.b,1,numSamples);
    end
    hAct{ll}=actF(hAct{ll});
end

ll = numHidden + 1;
y_hat = stack{ll}.W*hAct{ll-1}+repmat(stack{ll}.b,1,numSamples);
y_hat = exp(y_hat);
hAct{ll}=bsxfun(@rdivide,y_hat,sum(y_hat));
% predict probability 
pred_prob = hAct{ll}; 


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y_hat = log(hAct{numHidden+1});
index = sub2ind(size(y_hat),labels',1:numSamples);
fcost = -sum(y_hat(index)); % objective function cost

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% cross entropy gradient 
temp = zeros(size(hAct{numHidden+1}));
temp(index) = 1;
gradIn = hAct{numHidden+1} - temp; 
for ll = numHidden+1:-1:1
    if ll>numHidden
        dF = ones(size(gradIn));
    else
        dF = g_actF(hAct{ll});
    end
    
    dZ = gradIn.*dF;
    if ll>1 
        gradStack{ll}.W = dZ*hAct{ll-1}';
    else
        gradStack{ll}.W = dZ*data';
    end
    gradStack{ll}.b = sum(dZ,2);
    gradIn = stack{ll}.W'*dZ; 
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wcost =0; 
for ll=1:numHidden+1
    wcost = wcost + .5*ei.lambda*sum(stack{ll}.W(:).^2);
end

cost = fcost+wcost;

% compute the gradient of the weight decay 
for ll=1:numHidden 
    gradStack{ll}.W = gradStack{ll}.W + ei.lambda*stack{ll}.W;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



