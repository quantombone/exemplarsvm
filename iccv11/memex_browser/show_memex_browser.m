function allbbs = show_memex_browser(dataset_params, models, grid, ...
                                     fg, set_name, ...
                                     finalstruct, maxk)
% Show the exemplar-view of the memex browser
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%maxk is the maximum number of top detections we display
if ~exist('maxk','var')
  maxk = 5;
end

final_boxes = finalstruct.unclipped_boxes;
final_maxos = finalstruct.final_maxos;

bbs = cat(1,final_boxes{:});
imids = bbs(:,11);
exids = bbs(:,6);

moses = cat(1,final_maxos{:});

%% sort detections by score
[aa,bb] = sort(bbs(:,end), 'descend');
bbs = bbs(bb,:);

wwwdir = sprintf('%s/memex/%s.%s-%s%s/', dataset_params.localdir,...
                 set_name, models{1}.cls, ...
                 models{1}.models_name, finalstruct.calib_string);

if ~exist(wwwdir,'dir')
  mkdir(wwwdir);
end

%% show the exemplar view

filer = sprintf('%s/index.html', wwwdir);
fid = fopen(filer,'w');
fprintf(fid,['<html><head><title>memex browser</title>'...             
             '<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.1/jquery.min.js"></script>'...             
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/raphael.js"></script>'...
             '<script src="http://balaton.graphics.cs.cmu.edu/tmalisie/memex.js"></script>'...
             '</head><body>\n']);

fprintf(fid,'<table border=1>\n');
for i = 1:min(10,length(models))
  fprintf(fid,'<tr>\n');
  [a,curid,ext] = fileparts(models{i}.I);
  
  bb = models{i}.model.bb;

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
  
  goods = find(exids==i);
  for j = 1:maxk
    bb = bbs(goods(j),:);  

    bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                        '%d, %d, %d, %.3f, '...
                        '%d, %d, %d, %.3f]'],...
                       bb(1),bb(2),bb(3),bb(4),...
                       bb(5),bb(6),bb(7),bb(8),...
                       bb(9),bb(10),bb(11),bb(12));

    Isize = [grid{bb(11)}.imbb(4) grid{bb(11)}.imbb(3)];
    [a,curid,ext] = fileparts(fg{bb(11)});
    divid = sprintf('notepad%d.%d',i,j);
    fprintf(fid,'<td><div id="%s"/>',divid);
    fprintf(fid,'<script>show_image("%s","%s%s",%s,[%d,%d],"red");</script></td>',...
            divid, curid, ext, bbstring, Isize(1), Isize(2));

  end
  
  fprintf(fid,'</tr>\n');
  
end

fprintf(fid,'</table>\n');

fprintf(fid,'</body></html>\n');
fclose(fid);

