function model = learnGaussianTriggsAll(data_set, covstruct_full, ...
                                     add_flips, hg_size,A)

if ~exist('add_flips','var')
  add_flips = 1;
end

if ~exist('hg_size','var')
  hg_size = [];
end

if ~exist('A','var')
  A = [];
end

cls = cellfun2(@(x){x.objects.class},data_set);
cls = cat(2,cls{:});
ucls = unique(cls);

for i = 1:length(ucls)
  cls = ucls{i};
  m = learnGaussianTriggs(data_set,cls,covstruct_full,add_flips, ...
                          hg_size,A);
  model{i} = m;
end

clear m
m.data_set = data_set;
m.params = model{1}.params;

m.models = cellfun2(@(x)x.models{1},model(cellfun(@(x)length(x)>0,model)));
model = m;
