function scorevec = count_correct_person(models,grid,target_directory, ...
                              finalstruct)

allbbs = cell(0,1);
%final_boxes = final.final_boxes;
final_boxes = finalstruct.unclipped_boxes;
final_maxos = finalstruct.final_maxos;


VOCinit;

%if enabled we show images
saveimages = 0;

%% prune grid to contain only images from target_directory
[cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
                    ' %d']);

gridids = cellfun2(@(x)x.curid,grid);
goods = ismember(gridids,cur_set);
grid = grid(goods);

imids = cell(1,length(final_boxes));
for i = 1:length(final_boxes)
  imids{i} = [];
  if size(final_boxes{i},1) > 0
    imids{i} = i * ones(size(final_boxes{i},1),1);
    final_boxes{i}(:,5) = i;
  end
end

bbs = cat(1,final_boxes{:});
imids = cat(1,imids{:});
moses = [final_maxos{:}];
[aa,bb] = sort(bbs(:,end) .* double(moses'>.5),'descend');

N = sum(aa>0);
scorevec = zeros(N,2);
for i = 1:N
  imid = imids(bb(i));
  curbb = bbs(bb(i),:);
  
  curid = grid{imid}.curid;
  
  gt_has_person = has_person(curid,curbb);
  pred_has_person = has_person(models{curbb(1,6)}.curid, ...
                               models{curbb(1,6)}.gt_box);
  
  scorevec(i,:) = [gt_has_person pred_has_person];
  i
end



function p = has_person(curid,curbb)
p = 0;
VOCinit;
recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
I = imread(sprintf(VOCopts.imgpath,curid));

curp = find(ismember({recs.objects.class},{'person'}));
if length(curp) > 0
  currents = cat(1,recs.objects.bbox);
  currents = currents(curp,:);
  osmat = getosmatrix_bb(curbb,currents);
  if max(osmat)>.2
    p = 1;
  end
end
  
