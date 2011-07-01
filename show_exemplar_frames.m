function show_exemplar_frames(allmodels, N_PER_PAGE, dataset_params)
%% Draw the initialized exemplar frames as a 1x3 row of 3 images 
% Shows these 3 fields: input+gtbb+template, template mask+gtbb, HOG descriptor
% The visualization really shows what the template region is, and
% its relation to the ground-truth selection region

% if ~iscell(models)
%   m = models;
%   clear models;
%   models{1} = m;
% end

if ~exist('dataset_params','var')
  dataset_params = [];
end

pinds = do_partition(1:length(allmodels),N_PER_PAGE);
for i = 1:length(pinds)
  models = allmodels(pinds{i});

  figure(i)
  clf
  N = length(models);
  for i = 1:N
    m = models{i};
    o = (i-1)*3;
    subplot(N,3,o+1)
    I = convert_to_I(m.I);
    imagesc(I)
    plot_bbox(m.model.bb(1,:),'',...
              [1 0 0], [0 1 0], 0 ,[1 3],...
              m.model.hg_size)
    plot_bbox(m.gt_box,'',[0 0 1])
    axis image
    axis off
    title(sprintf('Ex %s.%d %s',m.curid,m.objectid,m.cls))
    
    subplot(N,3,o+2)
    onimage = m.model.mask*1;
    onimage(onimage==0) = 2;
    colors = [1 0 0; 0 0 1];
    cim = colors(onimage(:),:);
    cim = reshape(cim,[size(m.model.mask,1) size(m.model.mask,2) 3]);
    imagesc(cim);
    fullimbox = [0 0 size(cim,2) size(cim,1)]+.5;
    xform = find_xform(m.model.bb(1,1:4), fullimbox);
    gtprime = apply_xform(m.gt_box,xform);
    plot_bbox(fullimbox,'',...
              [1 0 0], [0 1 0], 0 ,[1 3],...
              m.model.hg_size)
    plot_bbox(gtprime,'',[0 0 1])
    axis image
    axis off
    grid on
    [u,v] = find(m.model.mask);
    
    curselection = [min(v) min(u) max(v) max(u)];
    curos = getosmatrix_bb(curselection, gtprime);
    
    title(sprintf('%s: Template:[%d x %d] \ncuros=%.2f Mask: [%d x %d]',...
                  m.model.init_params.init_type,...
                  m.model.hg_size(1),m.model.hg_size(2),...
                  curos,range(u)+1,range(v)+1));
    
    subplot(N,3,o+3)
    hogim = HOGpicture(repmat(m.model.mask,[1 1 features]).* ...
                       m.model.w);
    imagesc(hogim)
    axis image
    axis off
    grid on
    title('HOG features')
    drawnow
    
  end
end