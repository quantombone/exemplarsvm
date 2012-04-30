function obj = evaluate_obj(m)
%Return the SVM objective from linear SVM cost function with L2
%loss and L2 regularization (no bias regularization)
%Omega(w,b) = \frac{\lambda}{2}||w||^2 + \sum_{i}max(1-y_i*(w'*x_i+b),0)^2

posloss = sum(hinge(m.w(:)'*m.x-m.b));
negloss = sum(hinge(-(m.w(:)'*m.svxs-m.b)));
obj  = .5*m.w(:)'*m.w(:)/m.params.train_svm_c + ...
       (m.params.train_positives_constant* posloss + negloss);
