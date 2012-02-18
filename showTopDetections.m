function Isv = showTopDetections(data_set, boxes, L)
% Show top detections from the model

if ~exist('L','var')
  L = 100;
end

L = min(L,size(boxes,1));

mw = round(mean(boxes(:,4)-boxes(:,2)));
mh = round(mean(boxes(:,3)-boxes(:,1)));

PSIZE = round((mw+mh)/2*.4);


for i = 1:L
  fprintf(1,'.');
  b = boxes(i,:);

  I = toI(data_set{b(11)});
  I = pad_image(I,PSIZE);
  b = round(b + PSIZE);

  try
    crops{i} = I(b(2):b(4),b(1):b(3),:);
    crops{i} = imresize(crops{i},round([mw mh]),'bicubic');
    crops{i} = max(0.0,min(1.0,crops{i}));
  catch
    crops{i} = zeros(mw,mh,3);
  end
end

K1 = ceil(sqrt(L));
K2 = K1;
for j = (L+1):(K1*K2)
  crops{j} = zeros(round(mw),round(mh),3);
end

crops = reshape(crops,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,crops{j,:});
end

Isv = cat(1,svrows{:});

figure(2)
clf
imagesc(Isv)
drawnow

