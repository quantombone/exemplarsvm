function I=show_memex(A,ids,y)
if 0
for i = 1:length(ids)
  [a,b] = fileparts(ids{i});
  iconfile{i} = sprintf('/nfs/baikal/tmalisie/sun/icons/%s.png',b);  
  if ~fileexists(iconfile{i});
    I = convert_to_I(ids{i});
    I = max(0.0,min(1.0,imresize(I,[200 200])));
    imwrite(I,iconfile{i});
  end
end
end

uy = unique(y);
[aa,newy] = ismember(y,uy);
colors = jet(20);
r = randperm(20);
colors = colors(r(1:length(uy)),:);
other.colors = colors(newy,:);

A = A - diag(diag(A));
hits = find((sum(A,1) + sum(A,2)')>0);
%other.icon_string = @(x)sprintf('image="%s"',iconfile{hits(x)});

make_memex_graph(A(hits,hits),other);
I = make_memex_graph(A(hits,hits),other);

