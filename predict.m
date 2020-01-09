function [digit, odds] = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   digit = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
digit = zeros(size(X, 1), 1);

tmp = ones(size(X, 1), 1);
X = [tmp X];
a1 = [tmp sigmoid(X * Theta1')];

odds = sigmoid(a1 * Theta2');

[e, index] = max(odds, [], 2);
digit = index;

end
