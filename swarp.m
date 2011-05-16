function swarp(Is,bbs)

if ~exist('Is','var')
  %% generate fake data
  K = 9;
  for i = 1:K
    Is{i} = rand(400,400,3);
    b = ceil(rand(1,2)*100);
    rf = max(50,round(100*(randn(1,1)+1)));
    b = [b b+rf];
    bbs(i,:) = b;
    Is{i}(b(2):b(4),b(1):b(3),1:2)=0;
  end
end

K = length(Is);
KKK = ceil(sqrt(K));
figure(1)
clf
for i = 1:K
  subplot(KKK,KKK,i)
  imagesc(Is{i})
  plot_bbox(bbs(i,:));
  imbbs(i,:) = [1 1 size(Is{i},2) size(Is{i},1)];
  plot_bbox(imbbs(i,:),'',[0 0 0])
  axis image
  axis off
end


b = [400 400 600 600];
for i = 1:K
  xform = find_xform(bbs(i,:), b);
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
  
%   xform = find_xform(bbs(i,:), b);
%   result(i,:) = apply_xform(imbbs(i,:), xform);
%   d = result(i,:);
%   d([1 2]) = ceil(d([1 2]));
%   d([3 4]) = floor(d([3 4]));

%   xform = find_xform(b,bbs(i,:));
% ;
  
  
  clear zi
  for q = 1:3
    zi(:,q) = interp2(Is{i}(:,:,q),newpoints(:,1),newpoints(:,2));
  end

  zi(isnan(zi))=0;

  zi = reshape(zi,size(totalI,1),size(totalI,2),3);
  older = totalI;
  totalI = totalI + zi;

  %figure(2)

  %imshow(zi)
  %drawnow

  %keyboard
  imwrite(zi,sprintf('filer%05d.png',i));
end

totalI = totalI / size(result,1);


%totalI = max(0.0,min(1.0,totalI));
figure(2)
clf

imagesc(totalI)
plot_bbox(b,'center');
plot_bbox(result,'',[0 0 0]);
axis image
axis off
drawnow
