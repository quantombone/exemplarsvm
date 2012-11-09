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


if numel(hg_size) > 0 && numel(A) == 0
  lambda = .01;
  subinds = get_subinds(covstruct_full,hg_size);
  A = inv(lambda*eye(length(subinds)) + ...
          covstruct_full.c(subinds, ...
                           subinds));
end

cls = cellfun2(@(x){x.objects.class},data_set);
cls = cat(2,cls{:});
ucls = unique(cls);

for i = 1:length(ucls)
  cls = ucls{i};
  try
  m = learnGaussianTriggs(data_set,cls,covstruct_full,add_flips, ...
                          hg_size,A);
  catch
    m = [];
  end
  model{i} = m;
end

model = model(cellfun(@(x)length(x)>0,model));

clear m
m.data_set = data_set;
m.params = model{1}.params;

m.models = cellfun2(@(x)x.models{1},model(cellfun(@(x)length(x)>0,model)));
model = m;
