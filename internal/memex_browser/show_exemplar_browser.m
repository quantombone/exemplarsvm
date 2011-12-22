function show_exemplar_browser(dataset_params, models, ...
                               val_grid, val_set, ...
                               test_grid, test_set, ...
                               M, maxk)

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

val_maxos = cellfun2(@(x)x.extras.maxos,val_grid);
val_maxos = cat(1,val_maxos{:});

val_maxind = cellfun2(@(x)x.extras.maxind,val_grid);
val_maxind = cat(1,val_maxind{:});

val_maxclass = cellfun2(@(x)x.extras.maxclass,val_grid);
val_maxclass = cat(1,val_maxclass{:});

%get all bbs
test_bbs = cellfun2(@(x)x.bboxes,test_grid);
test_bbs = cat(1,test_bbs{:});

basedir = sprintf('%s/memex/',dataset_params.localdir);
wwwdir = [basedir '/' models{1}.models_name '/'];
if ~exist(wwwdir,'dir')
  fprintf(1,'making %s\n',wwwdir);
  mkdir(wwwdir);
end

filer = sprintf('%s/%s.html',...
                wwwdir, models{1}.cls);

fprintf(1,'Starting memex browser: %s\n', filer);

%%% show the index
%filer = sprintf('%s.index.html', wwwdir);
%wwwdir = [wwwdir '/stuff/'];

%if ~exist(wwwdir,'dir')
%  fprintf(1,'making %s\n',wwwdir);
%  mkdir(wwwdir);
%end

if fileexists(filer)
  fprintf(1,'already done, skipping html writing\n');
  return;
end
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
  bb(end) = 1.0;
  
  bb(1:4) = models{i}.gt_box(1:4);

  bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                      '%d, %d, %d, %.3f, '...
                      '%d, %d, %d, %.3f]'],...
                     bb(1),bb(2),bb(3),bb(4),...
                     bb(5),bb(6),bb(7),bb(8),...
                     bb(9),bb(10),bb(11),bb(12));

  Isize = models{i}.sizeI;
  divid = sprintf('exemplar.%d',i);
  divid2 = sprintf('exemplartitle.%d',i);
  fprintf(fid,'<td><table border=0>\n<tr><td><div id="%s" style="position:relative"></div></td></tr>\n<tr><td><div id="%s" style="position:relative"></div></td></tr></table>',divid,divid2);
  fprintf(fid,'<script>show_image("%s","%s","%s","%s","%d",%s,[%d,%d],"green");</script></td>',...
          divid, divid2, curid, ext, models{i}.objectid, bbstring, Isize(1), Isize(2));
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
  show_exemplar_page(wwwdir, models, i, dstring, val_bbs, val_set, ...
                     val_maxos, val_maxind, val_maxclass, ...
                     test_bbs, test_set, maxk);
end

function show_exemplar_page(wwwdir, models, i, dstring, val_bbs, ...
                            val_set, ...
                            val_maxos, val_maxind, val_maxclass, ...
                            test_bbs, test_set, maxk)



%Write html file header
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

%Draw the exemplar
bb = models{i}.model.bb;
bb(end) = 1.0;
bb(1:4) = models{i}.gt_box(1:4);

bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                    '%d, %d, %d, %.3f, '...
                    '%d, %d, %d, %.3f]'],...
                   bb(1),bb(2),bb(3),bb(4),...
                   bb(5),bb(6),bb(7),bb(8),...
                   bb(9),bb(10),bb(11),bb(12));

Isize = models{i}.sizeI;
%divid = sprintf('notepad%d.%d',i,0);
%fprintf(fid,'<td><div id="%s" style="position:relative"></div>', ...
%        divid);
divid = sprintf('exemplar.%d',i);
divid2 = sprintf('exemplartitle.%d',i);
fprintf(fid,['<td><table border=0><tr><td><div id="%s" style=' ...
             '"position:relative"></div></td></tr><tr><td><div id="%s" style="position:relative"></div></td></tr></table>'],divid,divid2);
  
fprintf(fid,'<script>show_image("%s","%s","%s","%s","%d",%s,[%d,%d],"green");</script></td>',...
        divid, divid2, curid, ext, models{i}.objectid, bbstring, Isize(1), Isize(2));

fprintf(fid,'</tr></table>\n<br/>\n');

fprintf(fid,'<h1>Trainval</h1>\n');
show_hit_table(fid, 'trainval', val_bbs, models, i, val_set, maxk, ...
               val_maxos, val_maxind, val_maxclass);
fprintf(fid,'<h1>Test</h1>\n');
show_hit_table(fid, 'test', test_bbs, models, i, test_set, maxk);

fprintf(fid,'</body></html>\n');
fclose(fid);

function show_hit_table(fid, prefix_string, bbs, models, i, cur_set, ...
                        maxk, maxos, maxind, maxclass)

if ~exist('maxos','var')
  maxos = zeros(size(bbs,1),1);
  maxind = maxos-1;
  maxclass = maxos*0-1;
end

MAX_ROWS_EXVIEW = 3;
%% sort detections by score in descending order
[aa,bb] = sort(bbs(:,end), 'descend');
bbs = bbs(bb,:);
exids = bbs(:,6);

maxos = maxos(bb);
maxind = maxind(bb);
maxclass = maxclass(bb);

goods = find(exids==i);


fprintf(fid,'<table border=1>\n');
fprintf(fid,'<tr>\n');

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
  
  %divid = sprintf('%s%d.%d',prefix_string,i,j);
  %fprintf(fid,'<td><div id="%s"/>',divid);
  
  divid = sprintf('%s.%d.%d',prefix_string,i,j);
  divid2 = sprintf('title.%s.%d.%d',prefix_string,i,j);
  fprintf(fid,'<td><table border=0><tr><td><div id="%s" style="position:relative"></div></td></tr><tr><td><div id="%s" style="position:relative"></div></td></tr></table>',divid,divid2);
  col = 'yellow';
  fprintf(fid,'<script>show_image_maxos("%s","%s","%s","%s","%d",%s,[%d,%d],"%s",%f,%d);</script></td>',...
          divid, divid2,curid, ext, maxind(goods(j)), bbstring, Isize(1), Isize(2),col,maxos(goods(j)),maxclass(goods(j)));
  
  if mod(j,MAX_ROWS_EXVIEW) == 0
    fprintf(fid,'</tr>\n<tr>\n');
  end
end

fprintf(fid,'</tr>\n');
fprintf(fid,'</table>\n');
