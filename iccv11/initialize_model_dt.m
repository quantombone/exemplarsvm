function model = initialize_model_dt(I,bbox,SBIN,hg_size)
%Get an initial model by cutting out a segment of a size which
%matches the bbox

warped = mywarppos(hg_size, I, SBIN, bbox);
curfeats = features(warped, SBIN);
model.x = curfeats(:);    
model.params.sbin = SBIN;

model.hg_size = size(curfeats);    
model.w = curfeats - mean(curfeats(:));
model.b = 0;

%%When doing dt, we should use bbox
%[model.target_id] = get_target_id(model,I);
%model.coarse_box = model.target_id.bb;
model.coarse_box = bbox;
