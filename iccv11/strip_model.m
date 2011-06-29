function models = strip_model(models)
%Take the models, and strip them of residual training data, only
%keep the information relevant for detection, this is useful for
%keeping a stripped version of models around for faster detections

for i = 1:length(models)
  cur = models{i};
  
  clear m;
  m.model.init_params = cur.model.init_params;
  m.model.hg_size = cur.model.hg_size;
  m.model.mask = cur.model.mask;
  m.model.w = cur.model.w;
  m.model.b = cur.model.b;
  m.model.bb = cur.model.bb;
  m.I = cur.I;
  m.curid = cur.curid;
  m.objectid = cur.objectid;
  m.cls = cur.cls;
  m.gt_box = cur.gt_box;
  m.sizeI = cur.sizeI;
  m.models_name = cur.models_name;
  
  models{i} = m;
  
  % models{i}.model.x = [];
  % models{i}.model.svxs = [];
  % models{i}.model.svbbs = [];
  
  % models{i}.model.wtrace = [];
  % models{i}.model.btrace = [];
  
  % models{i}.mining_stats = [];
  % models{i}.train_set = [];
end
