function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sumOfCost=0;
for itr=1:m,
	sig=sigmoid(theta'*X(itr,:)');
	y1=-y(itr)*log(sig);
	y0=-(1-y(itr))*(log(1-sig));
	sumOfCost=sumOfCost+y1+y0;
end

J=(1*sumOfCost)/m;

for j=1:size(theta,1),
	grad(j)=0;
	for itr1=1:m;
		sig=sigmoid(theta'*X(itr1,:)');
		grad(j)=grad(j)+((sig-y(itr1))*X(itr1,j));
	end
	grad(j)=grad(j)/m;
end




% =============================================================

end



