function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m,1) X]; % 5000 x 401 feature vector with first column = 1
a2 = [ones(m,1) sigmoid(a1 * Theta1')]; % 5000 x 401 feature matrix times 401 x 26 transposed weight matrix gives results in 5000 x 26 matrix, layer2 results + 1 column for each example
a3 = sigmoid(a2 * Theta2'); % 5000 x 26 results time 26 x 10 weight matrix gives 5000 x 10 probabilities for each class

[vals ind] = max(a3, [], 2);

p = ind % 5000 x 1 vector with class that had the highest prediction for the example row

% =========================================================================


end
