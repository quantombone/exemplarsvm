function [xs,bbs,Is] = get_global_wiggles(I)
%% Given an image, re-crop it and get features for those crops

scaler = 300/max(size(I,1),size(I,2));
I = imresize(I,scaler);

cut1 = round(linspace(1,round(size(I,1)/5),3));
cut2 = round(linspace(1,round(size(I,2)/5),3));

c = 1;
bbs = zeros(length(cut1)^4,4);
xs = zeros(1984,length(cut1)^4);
Is = cell(length(cut1)^4,1);
for i = 1:length(cut1)
  for j = 1:length(cut1)
    for k = 1:length(cut2)
      for l = 1:length(cut2)
        b = round([cut2(i) cut1(k) size(I,2)-cut2(j)+1 size(I,1)-cut1(l)+ ...
             1]);
        curI = I(b(2):b(4),b(1):b(3),:);
        curI = max(0.0,min(1.0,imresize(curI,[200 200])));
        x = features(curI,20);
        bbs(c,:) = b;
        xs(:,c) = x(:);
        Is{c} = curI;
        c = c + 1;
      end
    end
  end
end
