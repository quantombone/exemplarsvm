function show_memex_browser(dataset_params, models, val_grid, ...
                            val_set, test_grid, test_set, M, maxk)
% Show the exemplar-view of the memex browser
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%maxk is the maximum number of top detections we display
if ~exist('maxk','var')
  maxk = 30;
end

MAX_ROWS_INDEX = 3;


%get all bbs
val_bbs = cellfun2(@(x)x.bboxes,val_grid);
val_bbs = cat(1,val_bbs{:});

basedir = sprintf('%s/memex/',dataset_params.localdir);
finaldir = sprintf('/%s-%s%s/',...
                   models{1}.cls, ...
                   models{1}.models_name, '');

wwwdir = sprintf('%s/%s',basedir,finaldir);

fprintf(1,'Starting memex browser: %s\n',finaldir);

if ~exist(wwwdir,'dir')
  fprintf(1,'making %s\n',wwwdir);
  mkdir(wwwdir);
end

%%% show the index
filer = sprintf('%s/index.html', wwwdir);
fid = fopen(filer,'w');

fprintf(fid,['<html><head><title>memex browser</title>'...             
             '<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.1/jquery.min.js"></script>'...             
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/raphael.js"></script>'...
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/memex.js"></script>'...
             '</head><body>\n']);

%Write out the dataset string
dstring = sprintf('datadir = "http://balaton.graphics.cs.cmu.edu/tmalisie/%s/JPEGImages/";',...
                  dataset_params.dataset);
fprintf(fid,'<script>%s</script>\n',dstring);
fprintf(fid,'<h1>Exemplars cls="%s"</h1>\n', models{1}.cls);
fprintf(fid,'<table border=1>\n');
fprintf(fid,'<tr>\n');

for i = 1:length(models)
  [a,curid,ext] = fileparts(models{i}.I);
  bb = models{i}.model.bb;
  bb(1:4) = models{i}.gt_box(1:4);

  bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                      '%d, %d, %d, %.3f, '...
                      '%d, %d, %d, %.3f]'],...
                     bb(1),bb(2),bb(3),bb(4),...
                     bb(5),bb(6),bb(7),bb(8),...
                     bb(9),bb(10),bb(11),bb(12));

  Isize = models{i}.sizeI;
  divid = sprintf('notepad%d.%d',i,0);
  fprintf(fid,'<td><div id="%s" style="position:relative"></div>',divid);
  fprintf(fid,'<script>show_image_href("%s","%s%s",%s,[%d,%d],"green","%s.%d.html");</script></td>',...
          divid, curid, ext, bbstring, Isize(1), Isize(2),models{i}.curid, ...
          models{i}.objectid)
  if mod(i,MAX_ROWS_INDEX) == 0
    fprintf(fid,'</tr>\n<tr>\n');
  end
end

fprintf(fid,'</tr>\n');
fprintf(fid,'</table>\n');

fprintf(fid,'</body></html>\n');
fclose(fid);

%% show each exemplar's page
for i = 1:length(models)
  show_exemplar_page(wwwdir, models, i, dstring, val_bbs, val_set, maxk);
end

function show_exemplar_page(wwwdir, models, i, dstring, bbs, ...
                            cur_set, maxk)

MAX_ROWS_EXVIEW = 3;

%% sort detections by score in descending order
[aa,bb] = sort(bbs(:,end), 'descend');
bbs = bbs(bb,:);
exids = bbs(:,6);
imids = bbs(:,11);

filer = sprintf('%s/%s.%d.html', wwwdir, models{i}.curid, models{i}.objectid);
fid = fopen(filer,'w');
fprintf(fid,['<html><head><title>memex browser</title>'...             
             '<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.1/jquery.min.js"></script>'...             
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/raphael.js"></script>'...
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/memex.js"></script>'...
             '</head><body>\n']);
fprintf(fid,'<script>%s</script>\n',dstring);

fprintf(fid,'<table border=1>\n');
fprintf(fid,'<tr>\n');
[a,curid,ext] = fileparts(models{i}.I);

bb = models{i}.model.bb;
bb(1:4) = models{i}.gt_box(1:4);

bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                    '%d, %d, %d, %.3f, '...
                    '%d, %d, %d, %.3f]'],...
                   bb(1),bb(2),bb(3),bb(4),...
                   bb(5),bb(6),bb(7),bb(8),...
                   bb(9),bb(10),bb(11),bb(12));

Isize = models{i}.sizeI;
divid = sprintf('notepad%d.%d',i,0);
fprintf(fid,'<td><div id="%s" style="position:relative"></div>',divid);
fprintf(fid,'<script>show_image("%s","%s%s",%s,[%d,%d],"green");</script></td>',...
        divid, curid, ext, bbstring, Isize(1), Isize(2));

fprintf(fid,'</tr></table>\n<br/>\n');
fprintf(fid,'<table border=1>\n');
fprintf(fid,'<tr>\n');
goods = find(exids==i);

for j = 1:maxk
  bb = bbs(goods(j),:);  
  
  bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                      '%d, %d, %d, %.3f, '...
                      '%d, %d, %d, %.3f]'],...
                     bb(1),bb(2),bb(3),bb(4),...
                     bb(5),bb(6),bb(7),bb(8),...
                     bb(9),bb(10),bb(11),bb(12));
  
  [a,curid,ext] = fileparts(cur_set{bb(11)});
  curinfo = imfinfo(cur_set{bb(11)});
  Isize = [curinfo.Height curinfo.Width];
  
  divid = sprintf('notepad%d.%d',i,j);
  fprintf(fid,'<td><div id="%s"/>',divid);
  fprintf(fid,'<script>show_image("%s","%s%s",%s,[%d,%d],"red");</script></td>',...
          divid, curid, ext, bbstring, Isize(1), Isize(2));
  
  if mod(j,MAX_ROWS_EXVIEW) == 0
    fprintf(fid,'</tr>\n<tr>\n');
  end
end

fprintf(fid,'</tr>\n');
fprintf(fid,'</table>\n');

fprintf(fid,'</body></html>\n');
fclose(fid);

