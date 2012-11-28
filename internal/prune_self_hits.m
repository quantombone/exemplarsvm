function boxes = prune_self_hits(boxes,model)
imageids=cellfun(@(x)x.bb(1,11),model.models)';
bads = find(imageids(boxes(:,6))==boxes(:,11));
boxes(bads,:) = [];
