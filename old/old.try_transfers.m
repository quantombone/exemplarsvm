function try_transfers(models)
%Here is the NN segmentation transfer example

cls = models{1}.cls;
fg = get_pascal_bg('test',cls);

SBIN = models{1}.model.params.sbin;
hg_size = models{1}.model.hg_size;

xs = cellfun2(@(x)x.model.w(:),models);
xs = cat(2,xs{:});

VOCinit;

for i = 1:length(fg)
  I = convert_to_I(fg{i});
  [tmp,curid,tmp] = fileparts(fg{i});
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  classes = {recs.objects.class};
  goods = find(ismember(classes,{cls}));

  figure(1)
  clf
  for j = 1:length(goods)
    bb = recs.objects(goods(j)).bbox;
    %differ = round(20*rand(4,1))-10;
    %bb = bb + differ';
    m = initialize_model_dt(I,bb,SBIN,hg_size);
    x = m.x;
    
    subplot(1,5,1)
    imagesc(I);
    plot_bbox(bb);
    
    d = xs'*x;
    [aa,bb] = sort(d,'descend');
    for j = 1:4
      subplot(1,5,j+1)
      imagesc(get_exemplar_icon(models,bb(j)));
    end
    %% get nns

    drawnow
    pause(.1)
  end

end