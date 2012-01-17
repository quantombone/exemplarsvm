function extras = get_pascal_anno_function(dataset_params, Iname, boxes)
%Compute the "extra" information regarding the overlaps with the
%boxes inside this image

%extract image id from the filename (this can be done for VOC images)
%[tmp,curid,tmp] = fileparts(Iname);

% get GT objects for this image
[basedir,fname,ext] = fileparts(Iname);
gtfile = sprintf('%s/%s.xml',strrep(basedir,'JPEGImages', ...
                                'Annotations'),fname);
%gtfile = sprintf(dataset_params.annopath,curid);
recs = PASreadrecord(gtfile);

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
