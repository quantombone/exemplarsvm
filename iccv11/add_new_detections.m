function m = add_new_detections(m, xs, objids)
%% Add current detections (xs,objids) to the model struct (m)
%% (making sure we prune away duplicates, and then sort by score)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

m.model.nsv = cat(2,m.model.nsv,xs);
m.model.svids = cat(2,m.model.svids,objids);

names = cellfun2(@(x)sprintf('%d.%.3f.%d.%d.%d',x.curid,x.scale, ...
                             x.offset(1),x.offset(2),x.flip), ...
                 m.model.svids);


r = m.model.w(:)'*m.model.nsv;

[unames,subset,j] = unique(names);
m.model.svids = m.model.svids(subset);
m.model.nsv = m.model.nsv(:,subset);

[aa,bb] = sort(m.model.w(:)'*m.model.nsv,'descend');
m.model.svids = m.model.svids(bb);
m.model.nsv = m.model.nsv(:,bb);

