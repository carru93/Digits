clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

I = (255 -  double(imread(argv(){1}))(:)') ./ 255;

colormap(gray)

%subplot(1,2,1);
imshow(reshape(I, 20, 20), []);
% subplot(1,2,2);
% imshow(reshape(example, 20, 20), []);
pause;

[digit, odds] = predict(Theta1, Theta2, I);

odds = int8([mod(1:10, 10); odds .* 100]');
for i=1:10
  fprintf('\n%d \t %d%%', odds(i, 1), odds(i, 2));
endfor
fprintf('\nNeural Network Prediction: %d (digit %d)\n', digit, mod(digit, 10));

pause;
