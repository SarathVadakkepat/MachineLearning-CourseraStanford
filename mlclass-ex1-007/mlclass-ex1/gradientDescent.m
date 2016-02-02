function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	CostDer=[0;0];	
	CostDer_Th0=0;
	CostDer_Th1=0;

	for itr=1:m
		CostDer_Th0=CostDer_Th0 + (theta(1)+theta(2)*X(itr,2)-y(itr))*X(itr,1);
		CostDer_Th1=CostDer_Th1 + (theta(1)+theta(2)*X(itr,2)-y(itr))*X(itr,2);
	end
	CostDer=[CostDer_Th0;CostDer_Th1];
	theta=theta-(alpha*(CostDer/m));
	J_history(iter) = computeCost(X, y, theta);
end

end

