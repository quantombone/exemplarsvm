function m = cap_to_K_dets(m,K)
%Cap the detections inside m to only the top K dets

if size(m.model.nsv,2)<=K
  return
end
r = m.model.w(:)'*m.model.nsv - m.model.b;
[aa,bb] = sort(r,'descend');
goods = bb(1:K);
m.model.nsv = m.model.nsv(:,goods);
m.model.svids = m.model.svids(goods);