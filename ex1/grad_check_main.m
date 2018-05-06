addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled 

% for the linear regression check 
% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];

% Shuffle examples.
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1,1:400);
train.y = data(end,1:400);

% sample size 
M=size(train.X,2);
% the parameter dimension
N=size(train.X,1);

ntimes= 10; %  check how many times
num_checks = 10; 
options = struct('MaxIter', 1);
% check the looping implementation of linear regression 
theta = randn(N,1);  
 for ii = 1:ntimes
    errors = grad_check(@linear_regression,theta,num_checks,train.X,train.y)
    theta = minFunc(@linear_regression, theta, options, train.X, train.y);
 end
 
 % check the vectorized implementation of linear regression 
 theta = randn(N,1);  
 for ii = 1:ntimes
    errors = grad_check(@linear_regression_vec,theta,num_checks,train.X,train.y)
    theta = minFunc(@linear_regression_vec, theta, options, train.X, train.y);
 end
 
 % for the logistic regression check 
 % read data 
 binary_digits = true;
 [train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
 train.X = [ones(1,size(train.X,2)); train.X]; 
 test.X = [ones(1,size(test.X,2)); test.X];
 
 % sample size 
 M=size(train.X,2);
 % the parameter dimension 
 N=size(train.X,1);


 ntimes= 10; %  check how many times
 num_checks = 1; 
 
 % check the looping implementation of logistic regression 
 theta = randn(N,1);
 theta = minFunc(@logistic_regression, theta, struct('MaxIter', 100), train.X, train.y);
 for ii = 1:ntimes
    errors = grad_check(@logistic_regression,theta,num_checks,train.X,train.y)
    theta = minFunc(@logistic_regression, theta, options, train.X, train.y);
 end
 
 % check the vectorized implementation of logistic regression 
 theta = randn(N,1);
 theta = minFunc(@logistic_regression, theta, struct('MaxIter', 10), train.X, train.y);
 for ii = 1:ntimes
    errors = grad_check(@logistic_regression_vec,theta,num_checks,train.X,train.y)
    theta = minFunc(@logistic_regression_vec, theta, options, train.X, train.y);
 end
 
 
 