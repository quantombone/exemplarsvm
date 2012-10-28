function [allbb,alluv,alllvl,t] = pad_and_get_all_bb(t,hg_size,sbin)
%Extract all bounding boxes from the feature pyramid (and pad the pyramid)

allbb = cell(length(t.hog),1);
alluv = cell(length(t.hog),1);
alllvl= cell(length(t.hog),1);
for level = 1:length(t.hog)
  t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], ...
                          0);
  curids = zeros(size(t.hog{level},1),size(t.hog{level},2));
  curids = reshape(1:numel(curids),size(curids));

  goodids = curids(1:size(curids,1)-hg_size(1)+1,1:size(curids,2)- ...
                   hg_size(2)+1);

  [rawuuu,rawvvv] = ind2sub(size(curids),goodids(:));
  uuu = rawuuu - t.padder;
  vvv = rawvvv - t.padder;
  
  bb = ([vvv uuu vvv+hg_size(2) uuu+hg_size(1)] -1) * ...
       sbin/t.scales(level) + 1;
  bb(:,3:4) = bb(:,3:4) - 1;
  
  allbb{level} = bb;
  alluv{level} = [rawuuu rawvvv];
  alllvl{level} = goodids(:)*0+level;
end

alluv = cat(1,alluv{:});
allbb = cat(1,allbb{:});
alllvl = cat(1,alllvl{:});
allbb(:,5) = 0;
allbb(:,6) = 0;
allbb(:,7) = 0;
allbb(:,8) = t.scales(alllvl);
allbb(:,9) = alluv(:,1);
allbb(:,10) = alluv(:,2);
allbb(:,11) = 1;
allbb(:,12) = -1;

