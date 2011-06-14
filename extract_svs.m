function [target_id, target_x] = extract_svs(cb, N, set1, set2)
%Extract top N detection feature vectors from the feature
%pyramid. This allows us to only store (flip,scale,offset)
%information instead of the 8*8*31 numbers for the feature vector.
%To reconstruct the feature, we load the image, and follow the
%exact step of flip,scale,offset operations and only extract
%features from a single level of the pyramid (the target level)
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('set1','var')
  set1 = 'trainval';
  set2 = '';
end
if ~exist('set2','var')
  set2 = '';
end

%ONLY TAKE TRAINVAL HITS, no need for testing ones quite yet
trainval = get_pascal_bg(set1,set2);
ids = cellfun(@(x)get_file_id(x),trainval);
goods = find(ismember(cb(:,11),ids));

[tmp,bb] = sort(cb(goods,end),'descend');
bb = goods(bb);

if ~exist('N','var')
  N = 100;
end

if N > length(bb)
  N = length(bb);
end
fprintf(1,'Obtaining top N=%d feature vectors',N);

if nargout == 2
  target_x = zeros(8*8*31, N);
end
target_id = cell(1, N);
VOCinit;
tic
for i = 1:N
  fprintf(1,'.');

  target_id{i}.scale = cb(bb(i),8);
  target_id{i}.offset = cb(bb(i),9:10);
  target_id{i}.flip = cb(bb(i),7);
  target_id{i}.bb = cb(bb(i),1:4);
  target_id{i}.curid = cb(bb(i),11);
  %only do the feature extraction (which takes time!) when two
  %outputs are requested
  if nargout == 1
    continue
  end
  
  I = convert_to_I(sprintf(VOCopts.imgpath,...
                           sprintf('%06d',cb(bb(i),11))));

  if (target_id{i}.flip == 1)
    I = flip_image(I);
  end

  I = resize(I,target_id{i}.scale);
  full = features(I,8);
  f = padarray(full,[6 6 0]);

  f = f(target_id{i}.offset(1)+6+(0:7)-1,...
        target_id{i}.offset(2)+6+(0:7)-1,:);

  target_x(:,i) = f(:);
  %imagesc(HOGpicture(f))
  %differ=(m.model.w(:)'*f(:) - m.model.b) - cb(bb(i),end)

  %title(num2str(differ))
  %pause

  %cb(bb(i),end)
  % keyboard
end
toc
