function os = overlap_statistics(res)
VOCinit;

C = length(VOCopts.classes);
oscounts = zeros(C,C);

for i = 1:length(res.recs)
  curbb =   cat(1,res.recs(i).objects.bbox);
  os = getosmatrix_bb(curbb,curbb);
  os = os - diag(diag(os));

  [u,v] = find(os>.1);
  if length(u) == 0
    continue
  end
  goods = find(v>u);
  u = u(goods);
  v = v(goods);
  c = {res.recs(i).objects.class};
  [tmp,ids] = ismember(c,VOCopts.classes);
  curmat = sparse(ids(u(:)),ids(v(:)),1,C,C);

  oscounts = oscounts + curmat + curmat';
  
  fprintf(1,'.');
  
  
end
os = oscounts;

%os(15,:) = 0;
%os(:,15) = 0;
os = os - diag(diag(os));