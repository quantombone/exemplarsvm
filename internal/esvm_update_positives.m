function m = esvm_update_positives(m)
%Do latent SVM updates

if isfield(m,'models')
  m.models{1} = esvm_update_positives(m.models{1});
  return;
end

if ~isfield(m,'savex')
  return;
end

r = m.w(:)'*m.savex-m.b;
uhit = unique(m.savebb(:,6));
uhit = uhit(1:(length(uhit)/2));
curx = [];
curbb = [];
superinds = zeros(length(uhit),1);


for j = 1:length(uhit)
  goods = find(m.savebb(:,6)==uhit(j) | m.savebb(:,6)==(length(uhit)+ ...
                                                  uhit(j)));

  [aa,bb] = sort(r(goods),'descend');
  curx(:,end+1) = m.savex(:,goods(bb(1)));
  curbb(end+1,:) = m.savebb(goods(bb(1)),:);
  
end

m.x = curx;
m.bb = curbb;
