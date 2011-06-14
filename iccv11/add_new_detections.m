function m = add_new_detections(m, xs, bbs)
% Add current detections (xs,bbs) to the model struct (m)
% making sure we prune away duplicates, and then sort by score
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%First iteration might not have support vector information stored
if ~isfield(m.model, 'svxs')
  m.model.svxs = [];
  m.model.svbbs = [];
end

m.model.svxs = cat(2,m.model.svxs,xs);
m.model.svbbs = cat(1,m.model.svbbs,bbs);

%Create a unique string identifier for each of the supports
names = cell(size(m.model.svbbs,1),1);
for i = 1:length(names)
  bb = m.model.svbbs(i,:);
  names{i} = sprintf('%d.%.3f.%d.%d.%d',bb(11),bb(8), ...
                             bb(9),bb(10),bb(7));
end
  
[unames,subset,j] = unique(names);
m.model.svbbs = m.model.svbbs(subset,:);
m.model.svxs = m.model.svxs(:,subset);

[aa,bb] = sort(m.model.w(:)'*m.model.svxs,'descend');
m.model.svbbs = m.model.svbbs(bb,:);
m.model.svxs = m.model.svxs(:,bb);
