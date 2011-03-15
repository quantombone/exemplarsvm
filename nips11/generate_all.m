function models = generate_all(g)
startid = 202;

for i = 1:length(g{startid}.curw)
  startlvl = i;
  [w,b,ms,mos] = generate_iccv_graph(g,startid,startlvl);
  models{i}.model.w = w;
  models{i}.model.hg_size = size(w);
  models{i}.model.b = b;
  models{i}.model.ms = ms;
  models{i}.model.mos = mos;
  models{i}.gt_box = g{startid}.gt_box;
  models{i}.objectid = g{startid}.objectid;
  models{i}.curid = g{startid}.curid;
end