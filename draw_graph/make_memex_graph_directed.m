function [I]=make_memex_graph_directed(A, other)
%% Create a graph visualization
%% Tomasz Malisiewicz (tomasz@cmu.edu)
%% NOTE: A should be symmetric and have 1 largest component


if ~exist('other','var') || numel(other)==0
  other.is_silly = 1;
end

if ~isfield(other,'special_node')
  other.special_node = -1;
end

if ~isfield(other,'shapestring')
  for i = 1:size(A,1)
    other.shapestring{i} = 'shape=circle';
  end
end

if isfield(other,'evec_coloring')
  
  if 0
  A = A - diag(diag(A));
  degs = sum(A,1);
  L = normalized_laplacian(A);
  [V,D] = eig(full(L));
  D = D.*(D>0);
  [aa,bb] = sort(diag(D));
  V = V(:,bb);
  D = D(bb,bb);
  eres = diag(D);
  eres = eres(other.Kevec);
  volG = sum(degs);
  distsmat = sqrt(volG)*diag(degs.^-.5)*V;  

  dists = sqrt(volG)*diag(degs.^-.5)*V(:,other.Kevec);  
 
  end
  dists = other.V(other.Kevec,:);
  %[aa,bb] = ct_embedding(A, size(A,1));
  %dists2 = other.V(other.Kevec,:);
%keyboard

  
  NC = 200;
  colorsheet = jet(NC);
  colorsheet = colorsheet(end:-1:1,:);
  
  if abs(range(dists)) < .000001
    dists = dists*0+mean(dists);
  end
  
  
  dists = dists - min(dists);
  dists = dists / (max(dists)+eps);
  dists = round(dists*(NC-1)+1);
  
  %now dists are between 0 and 1
  other.colors = rgb2hsv(colorsheet(dists,:));
end

if ~isfield(other,'colors')
  %% generate white node colors
  other.colors = rgb2hsv(repmat([1 1 1],size(A,1),1));
end

if ~isfield(other, 'node_names');
  for i = 1:size(A,1)
    other.node_names{i} = '';%num2str(i);
  end
end

if ~isfield(other,'edge_names')
  other.edge_names = sparse_cell(size(A,1),size(A,2));
  [u,v] = find(A);
  for i = 1:length(u)
    other.edge_names{u(i),v(i)} = '';%sprintf('label="W=%.3f"',double(A(u(i),v(i))));
  end
end

if ~isfield(other,'edge_colors')
  
  [u,v] = find(A);
  other.edge_colors = sparse_cell(size(A,1),size(A,2));
  for i = 1:length(u)
    other.edge_colors{u(i),v(i)} = rgb2hsv([0 0 0]);
  end
    
  %other.edge_colors = colorsheet(dists,:);
  %other.edge_colors(:, 1) = 1;
end

if isfield(other,'ct_edge_coloring')
  fprintf(1,'got here\n');
  V = other.V;
  %V = ct_embedding(A, other.Kevec);
  [u,v] = find(A);
  dists = sum((V(1:other.Kevec,u) - V(1:other.Kevec,v)).^2,1);




  if 0
    figure(2)
    CT = getCTmatrix(A);
    CT = CT.*A;
    CT2 = distSqr_fast(V,V);
    CT2 = CT2.*A;
    imagesc(CT-CT2), colorbar
    title('difference in CTs')
    figure(1)
  end

  NC = 200;
  colorsheet = jet(NC);
  colorsheet = colorsheet(end:-1:1,:);
  
  if abs(range(dists)) < .000001
    dists = dists*0+mean(dists);
  end
    
  dists = dists - min(dists);
  dists = dists / (max(dists)+eps);
  dists = round(dists*(NC-1)+1);

  cur_colors = colorsheet(dists,:);
  
  other.edge_colors = sparse_cell(size(A,1),size(A,2));
  for i = 1:length(u)
    other.edge_colors{u(i),v(i)} = cur_colors(i,:);
  end
end

if ~isfield(other,'icon_string')
  other.icon_string = @(i)'';
end
  
for i = 1:size(A,1)
  other.colstring{i} = sprintf('fillcolor="%.3f %.3f %.3f"',...
                               other.colors(i,1), ...
                               other.colors(i,2),...
                               other.colors(i,3)); 
  
  other.node_names{i} = sprintf('label="%s"',other.node_names{i});
end


%% if this is turned on, then we do two step coloring
DO_COLORS = 0;
%A = A>0;
%A = (A+A')>0;

%% get largest connected component
%curA = A;
%curA(find(speye(size(curA)))) = 1;
%[p,q,r,s] = dmperm(curA);

%dr = diff(r);
%[aa,bb] = max(dr);
%inds = p(r(bb) : (r(bb+1)-1));
%A = curA(inds,inds);
%fprintf(1,'Largest CC has %d nodes\n',length(inds));

gv_file = '/nfs/hn22/tmalisie/ddip/memex.gv';
plain_file = '/nfs/hn22/tmalisie/ddip/memex.plain';
nodes_file = '/nfs/hn22/tmalisie/ddip/memex.nodes';

gv2_file = '/nfs/hn22/tmalisie/ddip/memex.2.gv';
ps_file = '/nfs/hn22/tmalisie/ddip/memex.ps';
png_file = '/nfs/hn22/tmalisie/ddip/memex.png';
if isfield(other,'pdf_file')
  pdf_file = other.pdf_file;
else
  pdf_file = '/nfs/hn22/tmalisie/ddip/memex.pdf';
end

if ~exist('special_node','var')
  special_node = -1;
end

%if ~exist('edge_names','var')
%  for i = 1:size(A,1)
%    edge_names{i} = sprintf('NODE %d',i);
%  end
%end

fprintf(1,'Dumping graph\n');
show_graph(A, gv_file, [], other);

if DO_COLORS == 1
  fprintf(1,'creating plain file\n');
  unix(sprintf('dot -Ksfdp -Tplain %s > %s', ...
               gv_file, plain_file));
  fprintf(1,'creating colors\n');
  unix(sprintf('grep node %s | awk ''{print($2,$3,$4)}'' > %s',...
               plain_file, nodes_file));
  
  r = load(nodes_file,'-ascii');
  positions = r(:,2:3);
  ids = r(:,1);
  [aa,bb] = sort(ids);
  positions = positions(bb,:);

  fprintf(1,'Dumping graph with colors\n');
  show_graph(A, gv2_file, positions);
else
  gv2_file = gv_file;
end

if nargout == 0
  fprintf(1,'creating pdf file %s\n', pdf_file);
  unix(sprintf('dot -Ksfdp -Tps2 %s > %s', ...
               gv2_file, ps_file));
  unix(sprintf('ps2pdf %s %s',ps_file,pdf_file));
else
  fprintf(1,'creating png file and loading\n');
  unix(sprintf('dot -Ksfdp -Tpng %s > %s', ...
               gv2_file, png_file));
  I = imread(png_file);
end

function show_graph(A, gv_file, positions, other)
[u,v] = find(A>0);
%goods = (v>=u);
%u = u(goods);
%v = v(goods);


fid = fopen(gv_file,'w');

fprintf(fid,'digraph G {\n');
fprintf(fid,['node [shape=circle style="filled" width=1.0 height=.5' ...
             ' penwidth=10 labelloc="t" fontsize="30"' ...
              ' labelfontcolor="black"]\n']);
fprintf(fid,'graph [outputorder="edgesfirst" size="20,20"]\n');
%fprintf(fid,'graph [page="8.5,11"]\n');
fprintf(fid,'edge [fontsize="10.0" penwidth=10 weight=10 arrowsize=5.0]\n');
%fprintf(fid,'bgcolor="black"\n');
fprintf(fid,'overlap="scale"\n');
%fprintf(fid,'fixedsize=true\n');

for i = 1:size(A,1)
 
  %if i == other.special_node
  %  shapestring = 'penwidth=50';%'style=filled fillcolor="red"';
  %end
  
  fprintf(fid,'%d [%s %s %s %s];\n',i,...
          other.shapestring{i},...
          other.colstring{i},...
          other.node_names{i},...
          other.icon_string(i));
  %end
  %fprintf(fid,'%d;\n',i);
end

if numel(positions) > 0
  dists = zeros(length(u),1);
  for i = 1:length(u) 
    dists(i) = norm(positions(u(i),:)-positions(v(i),:));
  end
  
  NC = 200;
  colorsheet = jet(NC);
  colorsheet = colorsheet(end:-1:1,:);
  
  
  dists = dists - min(dists);
  dists = dists / (max(dists)+eps);
  dists = round(dists*(NC-1)+1);
  
  %now dists are between 0 and 1

  edge_colors = colorsheet(dists,:);
end



for i = 1:length(u)
  other.edge_colors{u(i),v(i)} = rgb2hsv(other.edge_colors{u(i), ...
                    v(i)});
end

for i = 1:length(u)
  %if u(i)>v(i)
  %  continue
  %end
  
  fprintf(fid,'%d -> %d [weight=%.5f color="%.3f %.3f %.3f" %s];\n',...
          u(i), v(i), A(u(i),v(i)),...
          other.edge_colors{u(i),v(i)}(1),...
          other.edge_colors{u(i),v(i)}(2),...
          other.edge_colors{u(i),v(i)}(3), ...
          other.edge_names{u(i),v(i)});

end

fprintf(fid,'}\n');
fclose(fid);

