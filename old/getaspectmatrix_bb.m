function os = getaspectmatrix_bb(boxes,gts)
%% Given two sets of bounding boxes, N1 in the first one, and N2 in
%% the second one, compute a N1xN2 overlap score matrix where the
%% overlap score is the ratio of the intersection to union
%% Tomasz Malisiewicz (tomasz@cmu.edu)

W=boxes(:,3)-boxes(:,1)+1;
H=boxes(:,4)-boxes(:,2)+1;

theta = atan(W./H);

difftheta = repmat(theta,1,length(theta)) - repmat(theta,1,length(theta))';
os = abs(difftheta);
