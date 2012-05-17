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
             ' at %d\n'],size(m.svxs,2),10000);%m.params.train_max_negatives_in_cache);
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

if ~isfield(m,'savex')
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
        sun-size(olds, 1));
fprintf(1,'Updating Positives: #new obj = %.3f\n',...
        evaluate_obj(m));
