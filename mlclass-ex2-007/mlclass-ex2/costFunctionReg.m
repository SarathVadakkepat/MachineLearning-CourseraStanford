function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


sumOfCost=0;
for itr=1:m,
	sig=sigmoid(theta'*X(itr,:)');
	y1=-y(itr)*log(sig);
	y0=-(1-y(itr))*(log(1-sig));
	sumOfCost=sumOfCost+y1+y0;
end

J=(1*sumOfCost)/m;

sumOfTheta=0;
for j1=2:(size(theta,1)),
	sumOfTheta+=theta(j1)*theta(j1);
end

RegFactor= (lambda*sumOfTheta)/(2*m);


J=J+RegFactor;


for itr1=1:m;
		sig=sigmoid(theta'*X(itr1,:)');
		grad(1)=grad(1)+((sig-y(itr1))*X(itr1,1));
end
grad(1)=grad(1)/m;

for j=2:size(theta,1),
	grad(j)=0;
	for itr1=1:m;
		sig=sigmoid(theta'*X(itr1,:)');
		grad(j)=grad(j)+((sig-y(itr1))*X(itr1,j));
	end
	RegFactor=(lambda*theta(j))/m;
	grad(j)=(grad(j)/m)+RegFactor;
end



% =============================================================

end
