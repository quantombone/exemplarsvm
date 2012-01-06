function I = show_top_match_image(test_struct, models, test_set)

bbs = cat(1,test_struct.unclipped_boxes{:});
[aa,bb] = sort(bbs(:,end),'descend');
bbs = bbs(bb,:);
m = models{1};
m.model.svbbs = bbs;
try
m.model = rmfield(m.model,'svxs');
catch
end
m.train_set = test_set;

I = get_sv_stack(m,4,4);