function resc = get_canonical_bb(model, oldmodel)
%warp all bbs into a [100 100] box, so that we can warp all GTs
%into the same frame and compute overlap score between any two
%candidates

hg_size = model.params.init_params.hg_size;
center = [0 0 10*hg_size(1) 10*hg_size(2)];

resc = [];
for i = 1:length(model.models)
  bb = model.models{i}.bb;
  gt = model.models{i}.gt_box;
  
  for j = 1:size(bb,1)
    xform = find_xform(bb(j,:), center);
    transformed_c = apply_xform(gt, xform);
    resc(end+1,:) = transformed_c;
    
    %curos = getosmatrix_bb(gt,bb(j,:));
    %curos2 = getosmatrix_bb(transformed_c,center);

    % figure(1)
    % clf
    % imagesc(zeros(80,80,3))
    % plot_bbox(transformed_c)
    % axis image

    % figure(2)
    % clf
    % curI = toI(model.data_set(bb(j,11)));
    % imagesc(curI)
    % plot_bbox(bb(j,:))
    % plot_bbox(gt,'',[1 0 0])
    % drawnow
    % pause
    
  end
end
