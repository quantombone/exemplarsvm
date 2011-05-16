function m = add_new_detections(m, xs, objids)
%Add new detetions to the model, making sure we dont overcount duplicates
m.model.nsv = cat(2,m.model.nsv,xs);
m.model.svids = cat(2,m.model.svids,objids);

names = cellfun2(@(x)sprintf('%d.%d.%d.%d.%d',x.curid,x.level, ...
                             x.offset(1),x.offset(2),x.flip),m.model.svids);
r = m.model.w(:)'*m.model.nsv;

[unames,subset,j] = unique(names);
m.model.svids = m.model.svids(subset);
m.model.nsv = m.model.nsv(:,subset);
