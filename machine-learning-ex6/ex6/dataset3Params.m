function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
params = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
pn = size(params, 1);
results = zeros(pn);



for c_i = 1:pn
	for sig_i = 1:pn
		% Train the SVM
		model = svmTrain(X, y, params(c_i), @(x1, x2) gaussianKernel(x1, x2, params(sig_i)));
		% predict on cross validation set
		predictions = svmPredict(model, Xval);
		results(c_i, sig_i) = mean(double(predictions ~= yval))
	end
end


results
[max_c max_ci] = min(results)
[max_s max_si] = min(max_c)

C = params(max_ci(max_si))
sigma = params(max_si)


% =========================================================================

end
