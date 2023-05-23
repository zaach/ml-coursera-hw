function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

M = X * Theta'; % num_movies x num_users

interim = ((M - Y).^2 / 2) .* R;
reg_t = sum(sum((Theta .* Theta) * lambda / 2));
reg_x = sum(sum((X .* X) * lambda / 2));
J = sum(sum(interim)) + reg_t + reg_x;


rated_by_movie = sum(R, 2); % num_movies x 1
greg_x = lambda * (X .* 1);
X_grad = ((M - Y) .* R) * Theta + greg_x; % num_movies x num_features

rated_by_user = sum(R)'; % num_users x 1
greg_t = lambda * (Theta .* 1);
Theta_grad = ((M - Y) .* R)' * X + greg_t; % num_users x num_features


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
