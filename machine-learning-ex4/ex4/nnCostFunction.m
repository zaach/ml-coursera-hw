function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

K = (1:num_labels)'; % vector [1 2 ... K]
y_matrix = eye(num_labels)(y,:); % 5000 x 10 matrix of example classes
%X = [ones(m, 1) X]; % prepend 1 to X matrix

a1 = [ones(m,1) X]; % 5000 x 401 feature vector with first column = 1
z2 = a1 * Theta1'
a2 = [ones(m,1) sigmoid(z2)]; % 5000 x 401 feature matrix times 401 x 26 transposed weight matrix gives results in 5000 x 26 matrix, layer2 results + 1 column for each example
z3 = a2 * Theta2'
a3 = sigmoid(z3); % 5000 x 26 results time 26 x 10 transposed weight matrix gives 5000 x 10 probabilities for each class


h = a3'; % 26x5000
added = (log(h) * -y_matrix) - log(1-h) * (1-y_matrix); % 26x10

regular1 = sum(sum(Theta1(:,2:end).^2));
regular2 = sum(sum(Theta2(:,2:end).^2)); 
regular = lambda/(2*m) * (regular1 + regular2); 
J = trace(added) / m + regular; % trace gives the sum of the diag elements

%[vals ind] = max(a3, [], 2);
%p = ind % 5000 x 1 vector with class that had the highest prediction for the example row


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%regularGrad = theta * (lambda/m);
%regularGrad(1) = 0;
%grad = (((h-y)' * X) / m)' + regularGrad;

d3 = a3 - y_matrix % 5000 x 10
% 5000x10 * 10x25 .* 5000 x 25
% for each example, computes output errors times weights for each output node,
% times sigmoid val of inputs
%d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
d2 = d3 * Theta2(:,2:end) .* (a2(:,2:end) .* (1-a2(:,2:end)))

% 10x5000 * 5000x25
%Delta2 = [zeros(size(Theta2,1),1) d3' * a2(:,2:end)]
Delta2 = d3' * a2

% 400x5000 * 5000x25 
%Delta1 = (a1(:,2:end)' * d2)'
% 25 x 5000 * 5000x400
%Delta1 = [zeros(size(Theta1,1),1) d2' * a1(:,2:end)]
Delta1 = d2' * a1

D2_reg = lambda/m * Theta2;
D1_reg = lambda/m * Theta1;
D2_reg(:,1) = 0;
D1_reg(:,1) = 0;

lambda
D1_reg
D2_reg

Theta2_grad = Delta2 / m + D2_reg
Theta1_grad = Delta1 / m + D1_reg


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
