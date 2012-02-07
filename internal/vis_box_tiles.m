function [III,locations] = vis_box_tiles(Is,L)
%Given a bunch of images (all of same size) and a bb per image
%Create a visualization

if nargin == 1
  L = extract_bbs(Is);
end

uL = unique(L(:,11));

Is = Is(uL);

[tmp,ids] = ismember(L(:,11),uL);
L(:,11) = ids;

Is = cellfun2(@(x)toI(x),Is);
start_length = length(Is);
S = ceil(sqrt(length(Is)));
fillmore = S*S-length(Is);

Is = cat(1,Is,repmat({Is{1}*0+0},fillmore,1));
PAD = 10;
Is = cellfun2(@(x)pad_image(x,PAD,[1 1 1]),Is);

m1 = mean(cellfun(@(x)size(x,1),Is));
m2 = mean(cellfun(@(x)size(x,2),Is));
Is = cellfun2(@(x)max(0.0,min(1.0,imresize(x,round([m1 m2]), ...
                                                  'bicubic'))),Is);

Is = reshape(Is,[S S])';
for i = 1:S
  Irow{i} = cat(2,Is{i,:});
end

III = cat(1,Irow{:});


counter = 1;
locations = zeros(start_length,size(L,2));

for i = 1:S
  for j = 1:S
    if (counter > start_length)
      break;
    end
    cur = L(counter,:);
    cur([1 3]) = cur([1 3])+(j-1)*size(Is{1},2);
    cur([2 4]) = cur([2 4])+(i-1)*size(Is{1},1);
    cur = cur + PAD;
    
    locations(counter,:) = cur;
    counter = counter + 1;

  end
end


if nargout == 0
  imagesc(III)
  for i = 1:size(locations,1)
    plot_bbox(locations(i,:),num2str(locations(i,end)),[1 0 0],[1 0 ...
                    0])
  end
end

%L(:,end)