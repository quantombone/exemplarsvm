function Isv = showBoxes(data_set, boxes, MAX_BOXES, is_correct)
% Show top detection crops in one large image, where all crops are
% resized to the size of the average detection
% Input: 
%   data_set: (the set of images for which we have boxes)
%   boxes: a matrix of (Nx12) boxes where boxes(:,11) points to
%      imageid
%   MAX_BOXES: the maximum number of boxes to show (defaults to
%      100)
%   is_correct [optional] vector indicating whether the detection
%      was correct or not (0 for wrong AND 1 for correct)
% Output:
%   Isv: an image of top detections

if ~exist('MAX_BOXES','var')
  MAX_BOXES = 100;
end

MAX_BOXES = min(MAX_BOXES,size(boxes,1));
[aa,bb] = sort(boxes(:,end),'descend');
boxes = boxes(bb(1:MAX_BOXES),:);

if exist('is_correct','var')
  cols = [ 0 0 1; 1 0 0; 0 1 0];
  cols = cols(is_correct+2,:);      
  colors = cols(bb(1:MAX_BOXES),:);
end

mw = round(mean(boxes(:,4)-boxes(:,2)));
mh = round(mean(boxes(:,3)-boxes(:,1)));

factor = 100/max([mw mh]);
mw = round(mw*factor);
mh = round(mh*factor);

for i = 1:MAX_BOXES
  %fprintf(1,'.');
  b = boxes(i,:);

  I = toI(data_set{b(11)});
  b = round(b);

  us = b(2):b(4);
  vs = b(1):b(3);
  
  goodu = find(us>=1 & us<=size(I,1));
  goodv = find(vs>=1 & vs<=size(I,2));
  
  crops{i} = zeros(length(us),length(vs),3);
  crops{i}(goodu,goodv,:) = I(us(goodu),vs(goodv),:);
  if (size(crops{i},1)*size(crops{i},2))>0

    crops{i} = max(0.0,min(1.0,imresize(crops{i},round([mw mh]), ...
                                        'bicubic')));

  else
    crops{i} = zeros(mw,mh,3);
  end

  if exist('colors','var')
    crops{i} = pad_image(crops{i}, 3, colors(i,:));
  end
  
  % figure(1)
  % clf
  % imagesc(I)
  % if isfield(data_set{b(11)},'objects')
  %   objs = cat(1,data_set{b(11)}.objects.bbox);
  %   plot_bbox(objs)
  % end
  % c = [0 1 0];
  % if (is_correct(i) == 0)
  %   c = [1 0 0];
  % end
  % plot_bbox(boxes(i,:),'',c,c)
  % pause
  
  if b(7) == 1
    crops{i} = flip_image(crops{i});
  end
end

K1 = ceil(sqrt(MAX_BOXES));
K2 = K1;
for j = (MAX_BOXES+1):(K1*K2)
  crops{j} = zeros(round(mw),round(mh),3);
  if exist('colors','var')
    crops{j} = pad_image(crops{j},3,[1 1 1]);
  end
end

crops = reshape(crops,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,crops{j,:});
end
Isv = cat(1,svrows{:});

if nargout == 0
  figure(1)
  clf
  imagesc(Isv)
  drawnow
  Isv = [];
end

