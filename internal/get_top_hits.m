function tops = get_top_hits(model, hg_size)
%Given a stack of Exemplar-SVM model slices, find the best
%detection of size hg_size

if ~isfield(model,'w')

  template = ones(hg_size(1),hg_size(2));
  template = template / norm(template(:));
  rrr = (size(template)+1)/2;
  t2 = zeros(size(template)); 
  t2([floor(rrr(1)) ceil(rrr(1))],[floor(rrr(2)) ceil(rrr(2))]) ...
      = 1;
  template = template.*double(exp(-.1*bwdist(t2)));
else
  template = model.w;
end

PAD = 5;
for i = 1:length(model.models)
  models = model.models{i}.models;
  clear chosenfeats
  clear chosenmask
  clear chosenbb
  clear chosenval
  for q = 1:length(models)
    curf = models{q}.x;

    %mask = ones(size(curf,1),size(curf,2));
    mask = models{q}.mask;
    rrr = (size(mask)+1)/2;
    mask2 = zeros(size(mask)); 
    mask2([floor(rrr(1)) ceil(rrr(1))],[floor(rrr(2)) ceil(rrr(2))]) ...
        = 1;
    mask2 = double(exp(-.1*bwdist(mask2)));

    maskorig = mask;
    mask = mask.*mask2;
    mask = mask / norm(mask(:));
    
    mask = pad_image(mask,PAD);
    maskorig = pad_image(maskorig,PAD);
    curf = pad_image(curf,PAD);

    if isfield(model,'w')
      %once we have real features we do this
      r = fconvblas(curf, {template}, 1, 1);
      r = r{1};
      r = r - model.b;
    else
      r = conv2(mask,template,'valid');
    end
    %imagesc(r,[-1 1]);
    %drawnow

    [maxval,maxind] = max(r(:));
    [uu,vv] = ind2sub(size(r),maxind);

    chosenfeats{q} = curf(uu-1+(1:hg_size(1)),vv-1+(1:hg_size(2)),:);
    chosenmask{q} = maskorig(uu-1+(1:hg_size(1)),vv-1+(1:hg_size(2)),: ...
                         );
    curbb = models{q}.bb;
    D1 = (curbb(4)-curbb(2)+1)/models{q}.hg_size(1);
    D2 = (curbb(3)-curbb(1)+1)/models{q}.hg_size(2);
    ushift = uu-PAD-1;
    vshift = vv-PAD-1;
    
    curbbsave = curbb;
    flippy = curbb(7);

    
    if curbb(7) == 0
      curbb([1]) = curbb([1]) + vshift*D1;
      curbb([2]) = curbb([2]) + ushift*D1;
      curbb(3) = curbb(1) + hg_size(2)*D2 - 1;
      curbb(4) = curbb(2) + hg_size(1)*D1 - 1;
      %fprintf(1,'no flip code\n');
    else
      
      curbb([3]) = curbb([3]) - vshift*D1;
      curbb([4]) = curbb([4]) - ushift*D1;
      curbb(1) = curbb(3) - hg_size(2)*D2 - 1;
      curbb(2) = curbb(4) - hg_size(1)*D1 - 1;
      
      %fprintf(1,'flip code\n');

    end
    
    curbb(9) = curbb(9) + ushift;
    curbb(10) = curbb(10) + vshift;
    curbb(end) = maxval;
    chosenbb{q} = curbb;
    chosenval(q) = maxval;

    if 0
    figure(1)
    clf
    imagesc(toI(model.models{i}.I))

    plot_bbox(chosenbb{q},'',[1 0 0],[1 0 0],0,[1 1],hg_size);
    plot_bbox(models{q}.bb,'',[0 1 0],[0 1 0],1);
    axis image
    drawnow

    clear mmm
    mmm.models{1}.hg_size = hg_size;
    mmm.params = model.models{1}.params;
    c = chosenbb{q};
    c(11) = 1;
    [a,b]=esvm_reconstruct_features(c, mmm, ...
                                    {toI(model.models{i}.I)},1);

    figure(2)
    subplot(1,2,1)
    imagesc(HOGpicture(chosenfeats{q}-mean(chosenfeats{q}(:))));
    title('chosen feats')
    subplot(1,2,2)
    imagesc(HOGpicture(reshape(b-mean(b(:)),size(chosenfeats{q}))))
    title('reconstructed feats')
    drawnow
    norm(chosenfeats{q}(:)-b)
    pause
    end
  end
  
  [alpha,beta] = max(chosenval);
  
  tops{i}.x = chosenfeats{beta};
  tops{i}.bb = chosenbb{beta};
  tops{i}.mask = chosenmask{beta};
end