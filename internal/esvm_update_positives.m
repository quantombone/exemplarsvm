function m = esvm_update_positives(m,greedy,PSELECT,NASS,min_cache)
%Do latent SVM updates on the positive examples saved inside
%m.savex and m.savebb
% Inputs
%   m: the input model
%   [greedy]: do greedy assignment, defaults to [true]

if nargin>0 && isstr(m)
  return;
end

if ~exist('min_cache','var')
  min_cache = -1;
end

if size(m.svxs,2) < min_cache%m.params.train_max_negatives_in_cache
  fprintf(1,['Update positives: not updating cache_size=%d, ready' ...
             ' at %d\n'],size(m.svxs,2),min_cache);
  return;
  
end

if ~exist('PSELECT','var')
  PSELECT = .5;
  NASS = 3;
end

if ~exist('NASS','var')
  NASS = 3;
end

if ~exist('greedy','var')
  greedy = 1;
end

if rand > .8
  greedy = 1;
end

fprintf(1,'Updating Positives: greedy=%d\n',greedy);

if isfield(m,'models')
  m.models{1} = esvm_update_positives(m.models{1},greedy);
  return;
end

oldobj = evaluate_obj(m);
if isfield(m,'extra_models')
  
  m2.models = m.extra_models;
  m2.w = m.w;
  m2.b = m.b;
  
  tops = get_top_hits(m2,m.hg_size);

  
  m.x = cellfun2(@(x)x.x(:),tops);
  m.x = cat(2,m.x{:});
  
  m.mask = cellfun2(@(x)x.mask,tops);
  m.mask = cat(3,m.mask{:});
  m.mask = mean(m.mask,3);

  oldbb = m.bb;
  m.bb = cellfun2(@(x)x.bb,tops);
  m.bb = cat(1,m.bb{:});
  
  gtbb = cellfun2(@(x)x.gt_box,m.extra_models);
  gtbb = cat(1,gtbb{:});
  
  
  m.resc = get_canonical_bb(gtbb,m.bb,m.hg_size);  
  
    
  % if greedy ==0 && rand <= PSELECT
  %   ind = randperm(min(size(bb),NASS));
  %   K = 1;
  %   ind = ind(1:K);
  %   m.bb = m.bb(ind,:);
  %   m.oldbb = oldbb(ind,:);
  % end
  
  oses = diag(getosmatrix_bb(oldbb,m.bb));
  nchanged = sum(oses<.98);

  fprintf(1,'Updating Positives: #new elements = %d\n',...
          nchanged);
  fprintf(1,'Updating Positives: #old obj = %.3f\n,          --- new obj = %.3f\n',...
          oldobj,evaluate_obj(m));

  return;
end

if ~isfield(m,'savex')
  fprintf(1,'cannot do updates because savex is not present\n');
  return;
end

r = m.w(:)'*m.savex-m.b;
m.savebb(:,end) = r;
uhit = unique(m.savebb(:,6));
uhit = uhit(1:(length(uhit)/2));
curx = [];
curbb = [];
curc = [];
superinds = zeros(length(uhit),1);


for j = 1:length(uhit)
  goods = find(m.savebb(:,6)==uhit(j) | m.savebb(:,6)==(length(uhit)+ ...
                                                  uhit(j)));

  [aa,bb] = sort(r(goods),'descend');
  
  ind = 1;
  K = 1;
  if greedy ==0 && rand <= PSELECT
    ind = randperm(min(length(bb),NASS));
    ind = ind(1:K);
  end
  
  curx(:,end+1:end+K) = m.savex(:,goods(bb(ind)));
  curbb(end+1:end+K,:) = m.savebb(goods(bb(ind)),:);
  curc(end+1:end+K,:) = m.resc(goods(bb(ind)),:);
end

news = curbb(:,[1:4 7 11]);
olds = m.bb(:,[1:4 7 11]);

m.x = curx;
m.bb = curbb;
m.curc = curc;

sun = size(unique(cat(1,news,olds),'rows'),1);
fprintf(1,'Updating Positives: #new elements = %d\n',...
        abs(sun-size(olds, 1)));
fprintf(1,'Updating Positives: #old obj = %.3f\n,          --- new obj = %.3f\n',...
        oldobj,evaluate_obj(m));

