function [Z,V] = zca2(x)
epsilon = 1e-8;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
meanValue=mean(x,1);
x=x-repmat(meanValue,size(x,1),1);
xcov=x*x'/size(x,2);
[U,S,~]=svd(xcov);
V=U*diag(1./sqrt(diag(S)+epsilon))*U';
Z=V*x;
