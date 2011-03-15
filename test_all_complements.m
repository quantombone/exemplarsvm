function [results] = test_all_complements(models,final,grid,friend_classes,savegtids,saverecs)
VOCinit;

keep_thresh = 0.1;
max_os = 1.0;

for exid = 1:length(models)
  fprintf(1,'!');
  Iex = im2double(imread(sprintf(VOCopts.imgpath, ...
                                 models{exid}.curid)));
  models{exid}.sizeI = size(Iex);
end

%compute all friends

% classes = VOCopts.classes;

% countvec = zeros(1,20);
% for i = 1:length(models)
%   fprintf(1,'.');
%   recs = PASreadrecord(sprintf(VOCopts.annopath,models{i}.curid));
%   bbs = cat(1,recs.objects.bbox);
%   osmat = getosmatrix_bb(bbs(models{i}.objectid,:), bbs);  
%   c = {recs.objects.class};
%   goodname = ismember(c,friend_classes);
%   targets = find(osmat>keep_thresh & osmat <=max_os & goodname);
  
%   if length(targets) > 1
%     fprintf(1,'got multiple d, keeping best\n',length(targets));
%     [aa,bb] = sort(osmat(targets),'descend');
%     targets = targets(bb);
%     targets = targets(bb(1));
%   end
  
%   targetc = c(targets);

%   models{i}.friendbb = bbs(targets,:);
%   models{i}.friendc = targetc;
  
%   [aa,bb] = ismember(targetc,classes);
%   countvec(bb) = countvec(bb)+1;
% end

% frac_covered = countvec ./ length(models)*100;
%models_with_friends = models;

%friend_classes = find(countvec > 10);
%friend_classes = classes(friend_classes);

for z = 1:length(friend_classes)
  targetcls = friend_classes{z};
  
  models = cellfun2(@(x)initialize_with_friends(x,targetcls), ...
                    models);
  
  % for aaa = 1:length(models_with_friends)
  %   oks = find(ismember(models_with_friends{z}.friendc,targetcls));
  %   models{z}.friendbb = models_with_friends{z}.friendbb(oks,:);
  %   models{z}.friendc = models_with_friends{z}.friendc(oks);
  % end
 
bboxes = final.unclipped_boxes;

minoverlap = .5;
if 1 %~exist('minoverlap','var')
  fprintf(1,'transferring friends\n');
  %oldbboxes = bboxes;
  bboxes = transfer_friends(models, bboxes);
end

if 0
maxdet = cellfun2(@(x)max(x(:,end)),bboxes);
lens = cellfun(@(x)length(x),maxdet);
goods = find(lens>0);
[alpha,beta] = sort([maxdet{goods}],'descend');
for i = 1:length(beta)
  curid = grid{goods(beta(i))}.curid;
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  figure(1)
  clf
  imagesc(Ibase)
  plot_bbox(bboxes{goods(beta(i))});
  pause
end
end

for i = 1:length(bboxes)
  bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
end



filer=sprintf(VOCopts.detrespath,'comp3',targetcls);
fprintf(1,'Writing File %s\n',filer);
fid = fopen(filer,'w');

for i = 1:length(bboxes)
  curid = grid{i}.curid;
  for q = 1:size(bboxes{i},1)
    fprintf(fid,'%s %f %f %f %f %f\n',curid,...
            bboxes{i}(q,end), bboxes{i}(q,1:4));
  end
end
fclose(fid);

if exist('minoverlap','var')
  VOCopts.minoverlap = minoverlap;
end

[recall,prec,ap,apold] = VOCevaldet(VOCopts,'comp3',targetcls, ...
                                    true,savegtids,saverecs);
results{z}.recall = recall;
results{z}.prc = prec;
results{z}.ap = ap;
results{z}.apold = apold;
results{z}.targetcls = targetcls;

figure;
plot(recall,prec)
title(sprintf('%s to predict %s: apold=%.3f',models{1}.cls,targetcls,apold));
targetcls
drawnow
end

