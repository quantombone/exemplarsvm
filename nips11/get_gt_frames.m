function objids = get_gt_frames(objids)
VOCinit;


objidssave = objids;
if iscell(objids)
  objids = [objids{:}];
end

if isfield(objids(1),'extra')
  return;
end

overlaps = zeros(length(objids),1);
matchind = zeros(length(objids),1);
matchclass = zeros(length(objids),1);

curids = {objids.curid};
%load each image at once
uids = unique(curids);
VOCinit;

allbbs = cell(length(curids),1);
bg = get_pascal_bg('trainval');
for i = 1:length(uids)
  %[tmp,curid] = fileparts(bg{uids(i)});
  curid = uids{i};
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  gt_bb = cat(1,recs.objects.bbox);
  [tmp,curclass] = ismember({recs.objects.class},VOCopts.classes);

  
  hits = find(ismember(curids,uids{i}));

  startbb = cat(1,objids(hits).bb);


  %if ~isfield(m,'models_name') || strcmp(m.models_name,'dalal') ==
  %0
  %Ibase = imread(sprintf(VOCopts.imgpath,curid));
  resser = cell(size(startbb,1),1);
  for j = 1:size(startbb,1)
    xform = find_xform(startbb(j,:),[0 0 8 8]);
    
    curos = getosmatrix_bb(startbb(j,:),gt_bb);
    goods = (curos>0);
    if sum(goods) == 0
      %HACK, take first GT bb if really none overlap
      goods = 1;
    end
    [aa,bb] = sort(curos,'descend');
    curos = curos(bb);
    goods = goods(bb);
    goods = bb(find(goods));

    if length(goods) == 0
      resser{j} = [];
    end
    resser{j}.bbs = apply_xform(gt_bb(goods,:),xform);
    try
    resser{j}.bbs(:,5) = goods;
    catch
      keyboard
    end
   
    resser{j}.bbs(:,6) = curclass(goods);
    resser{j}.bbs(:,7) = curos(goods);

    % figure(1)
    % clf
    % imagesc(zeros(8,8,1))
    % plot_bbox([0 0 8 8]+.5,'',[0 0 1])
    % plot_bbox(resser{j}.bbs(1,:)+.5,'',[1 0 0])
    % axis([-2 10 -2 10])
    % drawnow
    % figure(2)
    % clf
    % imagesc(Ibase)
    % plot_bbox(gt_bb,'',[1 0 0]);
    % plot_bbox(startbb(j,:),'',[0 0 1])
    % axis image
    % pause
  end


  for q = 1:length(hits)
    objidssave{hits(q)}.extra = resser{q}.bbs;
  end
end

objids = objidssave;