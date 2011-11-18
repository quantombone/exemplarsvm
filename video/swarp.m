function swarp(Is,bbs)
%Is is a (N x 1) cell array of images
%bbs is a (N x 4) (or NxK with K>4 as long as bbs(:,1:4) is the bbs)

%writes a bunch of files with each bb aligned to itself in the center

filestring = 'combo%05d.png';

% if ~exist('Is','var')
%   %% generate fake data
%   K = 9;
%   for i = 1:K
%     Is{i} = rand(400,400,3);
%     b = ceil(rand(1,2)*100);
%     rf = max(50,round(100*(randn(1,1)+1)));
%     b = [b b+rf];
%     bbs(i,:) = b;
%     Is{i}(b(2):b(4),b(1):b(3),1:2)=0;
%   end
% end

Is = cellfun2(@(x)convert_to_I(x),Is);

b = mean(bbs,1);
W = b(3)-b(1)+1;
H = b(4)-b(2)+1;
factor = max(W,H)/100;
W = W*factor;
H = H*factor;

K = length(Is);

b = [400 400 400+W 400+H];
for i = 1:K
  xform = find_xform(bbs(i,:), b);
  imbbs(i,:) = [1 1 size(Is{i},2) size(Is{i},1)];
  result(i,:) = apply_xform(imbbs(i,:), xform);
end

MAXDIM = 1000;
min1 = min(result(:,1))
min2 = min(result(:,2));
o = [min1 min2 min1 min2]-1;
b = b - o;
result = result - repmat(o,size(result,1),1);

max1 = max(result(:,3));
max2 = max(result(:,4));

maxer1 = ceil(max1);
maxer2 = ceil(max2);
totalI = zeros(maxer2,maxer1,3);

[xxx,yyy] = meshgrid(1:size(totalI,2),1:size(totalI,1));
points = [xxx(:) yyy(:) xxx(:) yyy(:)];

for i = 1:size(result,1)
  %find xform from center box to original box
  xform = find_xform(b,bbs(i,:));
  newpoints = apply_xform(points,xform);
    
  clear zi
  for q = 1:3
    zi(:,q) = interp2(Is{i}(:,:,q),newpoints(:,1),newpoints(:,2));
  end

  zi(isnan(zi))=1;

  zi = reshape(zi,size(totalI,1),size(totalI,2),3);
  older = totalI;
  totalI = totalI + zi;

  %imwrite(zi,sprintf(filestring,i));
end

totalI = totalI / size(result,1);
figure(2)
clf

imagesc(totalI)
plot_bbox(b,'center');
%plot_bbox(result,'',[0 0 0]);
axis image
axis off
drawnow
