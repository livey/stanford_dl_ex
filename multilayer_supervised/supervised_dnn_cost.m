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
%%% YOUR CODE HERE %%%\
cost = 0 ; %% init cost function 
for ii=1:numHidden+1 % init gradStack.W and gradStack.b 
    if ii > 1
        prev_size = ei.layer_sizes(ii-1);
    else
        prev_size = ei.input_dim;
    end;
    cur_size = ei.layer_sizes(ii);
    
    gradStack{ii}.W =zeros(cur_size,prev_size);
    gradStack{ii}.b =zeros(cur_size,1);
end
[~,Ntrain]=size(data);
pred_prob =zeros(ei.output_dim,size(data,2));
for ii=1:Ntrain
    
    layer_in = data(:,ii);
    for jj=1:numHidden
        ztp1 = stack{jj}.W*layer_in+stack{jj}.b;
        layer_in =1./(1+exp(-ztp1));
        hAct{jj}=layer_in;
    end
    jj = jj+1;   %% the output layer using the soft-max cost function
    temp  = stack{jj}.W*layer_in+stack{jj}.b;
    layer_in = exp(temp);
    hAct{jj}= layer_in;
    prediction = layer_in/sum(layer_in);
    pred_prob(:,ii) =prediction;
    
    if ~po %  not only predict, then compute cost and gradient
        cost =cost - log(prediction(labels(ii)));
        %% back propagation to compute gradient
        delta = prediction;
        delta(labels(ii))=delta(labels(ii))-1;
        for jj=numHidden+1:-1:2
            gradStack{jj}.W = gradStack{jj}.W+ delta*hAct{jj-1}';
            gradStack{jj}.b = gradStack{jj}.b + delta;
            delta = stack{jj}.W'*delta.*(hAct{jj-1}.*(1-hAct{jj-1}));
        end
        jj=jj-1;% for the first layer
        gradStack{jj}.W = gradStack{jj}.W+ delta*data(:,ii)';
        gradStack{jj}.b = gradStack{jj}.b + delta;
    end
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% coimpute cost
%%% YOUR CODE HERE %%%
% M_out = hAct{numHidden};
% M_sum = sum(M_out);
% indx = sub2ind([ei.output_dim,Ntrain],labels,1:Ntrain);
% temp = M_out(indx)./M_sum;
% temp = log(temp);
% cost = -sum(temp);
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% for ii=1:Ntrain
%    temp = hAct{numHidden+1}(:,ii);
%    temp  =temp./sum(temp);
%    temp(labels(ii)) = temp(labels(ii))-1;
%    delta_pre= temp;
%    gradStack{numHidden+1}.W = gradStack{numHidden+1}.W+ delta_pre*hAct{numHidden+1}(:,ii);
%    gradStack{numHidden+1}.b= gradStack{numHidden+1}.b+delta_pre;
%    for jj=numHidden:-1:1
%        act = hAct{jj}(:,ii);
%        deri = act.*(1-act);
%        delta_pre = stack{jj}.W'*delta_pre.*deri;
%        gradStack{jj}.W = gradStack{jj}.W+ delta_pre*hAct{jj}(:,ii);
%        gradStack{jj}.b= gradStack{jj}.b+delta_pre;
%    end
% end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



