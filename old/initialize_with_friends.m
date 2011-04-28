function model = initialize_with_friends(model,target_class)
VOCinit;
recs = PASreadrecord(sprintf(VOCopts.annopath, model.curid));
friendboxes = cat(1, recs.objects.bbox);
os = getosmatrix_bb(model.gt_box, friendboxes);
friends = find(os>.1);
friends = setdiff(friends,model.objectid);
os = os(friends);

friendbb = friendboxes(friends,:);
friendclass = {recs.objects(friends).class};


hits = find(ismember(friendclass,target_class));
friendbb = friendbb(hits,:);
friendclass = friendclass(hits);
os = os(hits);
friends = friends(hits);


if length(friends) > 0
  fprintf(1,'Found %d friends, keeping all\n',length(friends));
  [aa,bb] = sort(os,'descend');
  hits = bb(1);
  %hits = bb;
  friendbb = friendbb(hits,:);
  friendclass = friendclass(hits);
  os = os(hits);
  friends = friends(hits);
end

model.friendbb = friendbb;  
model.friendclass = friendclass;
  
%if isfield(model,'FLIP_LR') && model.FLIP_LR == 1
%  model.friendbb = flip_box(model.friendbb,model.sizeI);
%end

