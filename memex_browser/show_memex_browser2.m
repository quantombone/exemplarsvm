function show_memex_browser2(dataset_params, models, res_struct, ...
                            fg, set_name, rc, maxk)
% Show the global "detection view" of the memex browser
%
% Tomasz Malisiewicz (tomasz@cmu.edu)
fprintf(1,'Starting memex browser\n');

set_name = [set_name res_struct.calib_string];

MAX_ROWS_INDEX = 3;
MAX_ROWS_EXVIEW = 3;

%get all bbs
bbs = cat(1,res_struct.unclipped_boxes{:});

  

%% sort detections by score in descending order
[aa,bb] = sort(bbs(:,end), 'descend');
bbs = bbs(bb,:);
exids = bbs(:,6);
imids = bbs(:,11);

if ~exist('rc','var') || length(rc)==0
  rc = -1*ones(size(bbs,1),1);
end

%%note if rc is present (results correct), then it is already sorted

sizers = cat(1,res_struct.imbb{:});
sizers = sizers(:,[4 3]);

%maxk is the maximum number of top detections we display
if ~exist('maxk','var') || length(maxk)==0
  maxk = min(size(bbs,1),100);
end

wwwdir = sprintf('%s/memex/%s.%s-%s%s', dataset_params.localdir,...
                 set_name, models{1}.cls, ...
                 models{1}.models_name, '');

if ~exist(wwwdir,'dir')
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

dstring = sprintf('datadir = "http://balaton.graphics.cs.cmu.edu/tmalisie/%s/JPEGImages/";',...
                  dataset_params.dataset);
fprintf(fid,'<script>%s</script>\n',dstring);

fprintf(fid,'<table border=1>\n');

for i = 1:maxk
  exid = bbs(i,6);
  [a,curid,ext] = fileparts(models{exid}.I);
  bb = models{exid}.model.bb;
  bb(1:4) = models{exid}.gt_box(1:4);

  bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                      '%d, %d, %d, %.3f, '...
                      '%d, %d, %d, %.3f]'],...
                     bb(1),bb(2),bb(3),bb(4),...
                     bb(5),bb(6),bb(7),bb(8),...
                     bb(9),bb(10),bb(11),bb(12));

  Isize = models{exid}.sizeI;
  divid = sprintf('notepad%d.%d',i,0);
 
  fprintf(fid,'<tr><td><div id="%s" style="position:relative"></div>',divid);
  fprintf(fid,'<script>show_image("%s","%s%s",%s,[%d,%d],"green");</script></td>',...
          divid, curid, ext, bbstring, Isize(1), Isize(2));

  
  bb = bbs(i,:);  
  
  bbstring = sprintf(['[%.3f, %.3f, %.3f, %.3f, ',...
                      '%d, %d, %d, %.3f, '...
                      '%d, %d, %d, %.3f]'],...
                     bb(1),bb(2),bb(3),bb(4),...
                     bb(5),bb(6),bb(7),bb(8),...
                     bb(9),bb(10),bb(11),bb(12));
  
  
  [a,curid,ext] = fileparts(fg{bb(11)});
  
  curinfo = imfinfo(fg{bb(11)});
  Isize = [curinfo.Height curinfo.Width];
 
   
  if rc(i) == 1
    col = 'yellow';
  else
    col = 'red';
  end
  fprintf(1,'rc is %d\n',rc(i));
  
  divid = sprintf('notepad%d.%d',i,1);
  fprintf(fid,'<td><div id="%s"/>',divid);
  fprintf(fid,'<script>show_image("%s","%s%s",%s,[%d,%d],"%s");</script></td>',...
          divid, curid, ext, bbstring, Isize(1), Isize(2),col);
  
  fprintf(fid,'</tr>\n');
  
end

fprintf(fid,'</tr>\n');
fprintf(fid,'</table>\n');

fprintf(fid,'</body></html>\n');
fclose(fid);

