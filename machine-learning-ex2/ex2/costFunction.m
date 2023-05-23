function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%J = (((X * theta)-y) .^ 2)' * ones(m, 1) / (2 * m);

h = sigmoid(X * theta);
added = (log(h) .* -y) - log(1-h) .* (1-y);
J = added' * ones(m, 1) / m;

%theta = theta - ((((X * theta)-y)' * X)*alpha / m)';

grad = (((h-y)' * X) / m)';
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
