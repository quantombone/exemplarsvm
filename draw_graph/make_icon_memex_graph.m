function I=make_icon_memex_graph(subA, train, ids, other)

%subA = A(ids,ids);

if ~exist('other','var')
  other.created = 1;
end
colors = rgb2hsv(color_sheet);
other.colors = colors(train.info.classes(ids),:);

other.icon_string = @(i)sprintf('image="%s"',labelme_geticon_name(train, ...
                                                  ids(i)));

%other.icon_string = @(i)'';
                           
other.special_node = -1;

for i = 1:length(ids)

  other.shapestring{i} = sprintf('shape=circle URL="http://balaton.graphics.cs.cmu.edu/tmalisie/onega/memex/labelme400/www/largevis/%d.pdf"', ...
                                 ids(i));
  %other.shapestring{i} = 'shape=circle URL="http://www.google.com"';
end



for i = 1:length(ids)
  other.node_names{i} = '';%num2str(i);
end

uc = unique(train.info.classes(ids));
for i = 1:length(uc)
  if uc(i)==train.info.classes(ids(1))
    hit = find(train.info.classes(ids)==uc(i),2,'first');
  else
    hit = find(train.info.classes(ids)==uc(i),1,'first');
  end
  for q = 1:length(hit)
    other.node_names{hit(q)} = train.info.classnames{uc(i)};
  end
end

other.node_names{1} = ['START=' other.node_names{1}];

if 0
%% make some edge_colors
[u,v] = find(subA);
inds = find(subA);

[CT,Q,G] = getCTmatrix(subA);
%dists = G(inds);
dists = -CT(inds);
%dists = subA(inds);


NC = 200;
colorsheet = jet(NC);
colorsheet = colorsheet(end:-1:1,:);
    
dists = dists - min(dists);
dists = dists / (max(dists)+eps);
dists = round(dists*(NC-1)+1);

%now dists are between 0 and 1
other.edge_colors = colorsheet(dists,:);
end

%other.pdf_file = sprintf('/nfs/baikal/tmalisie/labelme400/www/largevis/%d.pdf',ids(1));
fprintf(1,'Calling mmg\n');
if nargout == 0
  make_memex_graph(subA, other);
else
  I = make_memex_graph(subA, other);
end
%train,ids,A,node_names,edge_names,special_node)
