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

J = sum(sum(((X*Theta'-Y).*R).^2))/2;
J = J + (lambda/2)*sum(sum(Theta.^2))+ (lambda/2)*sum(sum(X.^2));

for i=1:num_movies
##  %Method 1: not vectorized
##  for k=1:num_features
##    X_grad(i,k) = (X(i,:)*Theta'-Y(i,:)).*R(i,:)*Theta(:,k);
##  endfor

  % Method 2: vectorized
  X_grad(i,:) = (X(i,:)*Theta'-Y(i,:)).*R(i,:)*Theta;
  
##  % Method 3: vectorized, as suggested in instruction
##  idx = find(R(i,:)==1); %the ids of users that has rated to movie i
##  Theta_tmp = Theta(idx,:);
##  Y_tmp = Y(i,idx);
##  X_grad(i,:) = (X(i,:)*Theta_tmp'-Y_tmp)*Theta_tmp;
endfor

X_grad = X_grad + (lambda)*X;

for j=1:num_users
##  %Method 1: not vectorized
##  for k=1:num_features
##    Theta_grad(j,k) = ((X*Theta(j,:)'-Y(:,j)).*R(:,j))'*X(:,k);
##  endfor

  %Method 2: vectorized
  Theta_grad(j,:) = ((X*Theta(j,:)'-Y(:,j)).*R(:,j))'*X;
  
##  % Method 3: vectorized, as suggested in instruction
##  idx = find(R(:,j)==1); %the movie ids that user j has rated to
##  X_tmp = X(idx,:);
##  Y_tmp = Y(idx,j);
##  Theta_grad(j,:) = (X_tmp*Theta(j,:)'-Y_tmp)'*X_tmp;
endfor

Theta_grad = Theta_grad + (lambda)*Theta;
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
