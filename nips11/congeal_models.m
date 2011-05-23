function congeal_models(models)
%Congeal the models

%get all xs
K = 1;
xs = cellfun2(@(x)x.model.x(:,1:K),models);
xs = cat(2,xs{:});

for i = 1:length(models)
  fprintf(1,'.');
  bbs = cellfun2(@(x)x.bb,models{i}.model.target_id(1:K));
  bbs = cat(1,bbs{:});
  gts = repmat(models{i}.gt_box,size(bbs,1),1);
  b{i} = bbs;
  g{i} = gts;
  ind{i} = gts(:,1)*0 + i;
  ind2{i} = (1:size(gts,1))';
end

bs = cat(1,b{:});
gs = cat(1,g{:});
inds = cat(1,ind{:});
inds2 = cat(1,ind2{:});

c_box = [1 1 10 10];

for i = 1:size(bs,1)
  xform = find_xform(bs(i,:),c_box);
  newb = apply_xform(gs(i,:),xform);
  %newb = newb + randn(size(newb));
  if 0
    figure(1)
    clf
    imagesc(ones(10,10,3));
    plot_bbox(newb)
    axis image
    axis off
    drawnow
    pause
  end
  res(i,:) = newb;
end

d = distSqr_fast(xs,xs);
d = exp(-d/mean(d(:)));
os = getosmatrix_bb(res,res);

xs2 = xs*0;
for i = 1:size(xs,2), 

  c = reshape(xs(:,i),models{1}.model.hg_size);
  mask = repmat(sum(c.^2,3)>0,[1 1 features]);
  xs2(mask,i) = xs(mask,i) - mean(xs(mask,i)); 
  xs2(:,i) = xs2(:,i).*(xs2(:,i)>0);
end

d2 = (exp(.1*xs2'*xs));
d = d2.*os;


rt = randperm(size(d,1));
for targeti = 1:1000
  target = rt(targeti);
  [aa,bb] = sort(d(:,target),'descend');

  clear res
  res = cell(0,1);
  current = 1;
  oldinds = [];
  
  NNN = 3;
  NNN2 = NNN*NNN
  
  while length(res) < NNN2
    if sum(ismember(oldinds,inds(bb(current)))) > 0
      current = current + 1;
      continue
    end
    [a,b,c] = get_exemplar_icon(models,...
                                inds(bb(current)),...
                                0,inds2(bb(current)));
    
    res{end+1} = c;
    
    oldinds = [oldinds inds(bb(current))];
    current = current + 1;

  end
  

  figure(1)
  clf
  for i = 1:NNN2
    subplot(NNN,NNN,i)
    imagesc(res{i})
  end


  
  drawnow
  m1 = round(mean(cellfun(@(x)size(x,1),res)));
  m2 = round(mean(cellfun(@(x)size(x,2),res)));
  
  res2 = cellfun2(@(x)max(0.0,min(1.0,imresize(x,[m1 m2]))),res);
  res2 = cat(4,res2{:});
  res2 = mean(res2,4);
  figure(2)
  imagesc(res2)
  drawnow
  pause
  
  drawnow  
  
end
