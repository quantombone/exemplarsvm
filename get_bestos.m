function bestos = get_bestos(bbox2,bbox)
%get best possible overlap between two bounding boxes
%by trying to align bbox2 onto bbox
W1 = bbox(3)-bbox(1)+1;
H1 = bbox(4)-bbox(2)+1;

starts = bbox(:,[1 2]);
bbox(:,1) = bbox(:,1) - starts(1) + 1;
bbox(:,3) = bbox(:,3) - starts(1) + 1;
bbox(:,2) = bbox(:,2) - starts(2) + 1;
bbox(:,4) = bbox(:,4) - starts(2) + 1;


W2 = bbox2(3)-bbox2(1)+1;
H2 = bbox2(4)-bbox2(2)+1;

starts = bbox2(:,[1 2]);
bbox2(:,1) = bbox2(:,1) - starts(1) + 1;
bbox2(:,3) = bbox2(:,3) - starts(1) + 1;
bbox2(:,2) = bbox2(:,2) - starts(2) + 1;
bbox2(:,4) = bbox2(:,4) - starts(2) + 1;



scale1 = H2/H1;
scale2 = W2/W1;

mins = min(scale1,scale2);
maxs = max(scale1,scale2);

scales = linspace(max(.001,mins-.5),maxs+.5,1000);
bboxes = zeros(length(scales),4);
bboxes(:,[1 2]) = 1;
bboxes(:,[3 4]) =  [scales'*W1 scales'*H1];

os = getosmatrix_bb(bbox2,bboxes);
bestos = max(os);


  

