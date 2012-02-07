function obj = evaluate_obj(m)
%Return the SVM objective from libsvm

p = m.params;
posloss = sum(hinge(m.w(:)'*m.x-m.b));
negloss = sum(hinge(-m.w(:)'*m.svxs+m.b));
obj  = .5*m.w(:)'*m.w(:) + p.train_svm_c * ...
       (p.train_positives_constant* posloss + negloss);

function r = hinge(x)
r = max(1-x,0);