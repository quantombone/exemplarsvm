function extras = get_pascal_anno_function(dataset_params, Iname, boxes)

[tmp,curid,tmp] = fileparts(Iname);

% get GT objects for this image
recs = PASreadrecord(sprintf(dataset_params.annopath,curid));

% get overlaps with all ground-truths (makes sense for VOC
% images only)
gtbb = cat(1,recs.objects.bbox);
os = getosmatrix_bb(boxes,gtbb);
cats = {recs.objects.class};
[tmp,cats] = ismember(cats,dataset_params.classes);

[alpha,beta] = max(os,[],2);
extras.maxos = alpha;
extras.maxind = beta;
extras.maxclass = reshape(cats(beta),size(beta));
