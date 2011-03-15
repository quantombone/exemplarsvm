function label_buses%(models,uhits)


%models = load_all_models('bus','exemplars','10');
%uhits = 1:length(models);

%%Here is the script for labeling buses
VOCinit;

cls = 'bus';
fg = get_pascal_bg('trainval',cls);

resdir = '/nfs/baikal/tmalisie/buslabeling/';

for i = 1:length(fg) %uhits)

  %curid = models{uhits(i)}.curid;
  %objectid = models{uhits(i)}.objectid;
  %I = imread(sprintf(VOCopts.imgpath, curid));
  %I = im2double(I);
  
  I = convert_to_I(fg{i});
  [tmp,curid] = fileparts(fg{i});
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  
  ids = find(ismember({recs.objects.class},cls));
  fprintf(1,'Found %d instances here\n',length(ids));
  for q = 1:length(ids)
    objectid = ids(q);
    
    resfile = sprintf('%s/%s.%d.mat',resdir,curid,objectid);
    showfile = sprintf('%s/%s.%d.png',resdir,curid,objectid);
    if fileexists(resfile)
      fprintf(1,'%s exists\n',resfile);
      continue;
    end
    curobj = recs.objects(objectid);
    bb = curobj.bbox;
    curI = I*0;
    PAD = 10;
    bb(1:2) = bb(1:2)-PAD;
    bb(3:4) = bb(3:4)+PAD;
    bb(1:2) = max(1,bb(1:2));
    bb(3) = min(bb(3),size(I,2));
    bb(4) = min(bb(4),size(I,1));

    curI(bb(2):bb(4),bb(1):bb(3),:) = I(bb(2):bb(4),bb(1):bb(3),:);
    
    label_instance(I,curI,bb,showfile,resfile);
    
    
  end
  %pos = poly2mask(pos(:,1),pos(:,2),size(I,1),size(I,2));
  %keyboard
end



function [shower,seg,faces] = build_segmentation(I,f,b,l,r,t)

faces = {f,b,l,r,t};

allpoints = cat(1,faces{:});
ids = [];
for i = 1:length(faces)
  for j = 1:size(faces{i},1)
   ids(end+1) = i; 
  end
end
d = distSqr_fast(allpoints',allpoints');
same = (sqrt(d)<10);

[p,q,r,s] = dmperm(same);

components = [];
for i = 1:length(r)-1
  components{end+1} =  p(r(i):r(i+1)-1);
end

for i = 1:length(components)
  meaner = mean(allpoints(components{i},:),1);
  for j = 1:length(components{i})
    allpoints(components{i}(j),:) = meaner;
  end
end

uc = unique(ids);
for i=1:length(uc)
  faces{uc(i)} = allpoints(ids==uc(i),:);
end

seg = zeros(size(I,1),size(I,2),1);
for i = 1:length(faces)
  if prod(size(faces{i}))==0
    continue
  end
  curpoly = poly2mask(faces{i}(:,1),faces{i}(:,2),size(I,1),size(I, ...
                                                  2));
  seg(curpoly==1)=i;
end

colors = jet(5);
colors(end+1,:) = 0;
colors = colors(end:-1:1,:);
seg2 = colors(seg(:)+1,:);
seg2 = reshape(seg2,size(I));

fg = repmat((seg>0),[1 1 3]);
shower = I;
shower(find(fg)) = shower(find(fg))*.5 + .5*seg2(find(fg));

function label_instance(I,curI,bb,showfile,resfile)

    isdone = 0;
    while isdone == 0
      
      figure(1)
      imagesc(curI)      
      f = zeros(0,2);
      b = zeros(0,2);
      l = zeros(0,2);
      r = zeros(0,2);
      t = [];
      lines = cell(0,1);
      while 1
        str = input(['(c)lear, (f)ront, (b)ack, (l)eft, (r)ear,' ...
                     ' (t)op, (n)ext]'],'s');
        if strcmp(str,'c')
          fprintf(1,'Clearing and restarting\n');
          break;
        elseif strcmp(str,'n')
          [shower,seg,faces] = build_segmentation(I,f,b,l,r,t);
          figure(2)
          imagesc(shower)
          fprintf(1,'Continuing');
          isdone = 1;
          clear res
          res.seg = seg;
          res.faces = faces;
          imwrite(shower,showfile);
          save(resfile,'res');
          break;
        %elseif strcmp(str,'l')
        %  h = imline;
        %  res = iptgetapi(h);
        %  lines{end+1} = res.getPosition();
        elseif strcmp(str,'f')
          fprintf(1,'Label Front \n');
          h = impoly;
          res = iptgetapi(h);
          f = res.getPosition();
        elseif strcmp(str,'b')
          fprintf(1,'Label Back \n');
          h = impoly;
          res = iptgetapi(h);
          b = res.getPosition();
        elseif strcmp(str,'l')
          fprintf(1,'Label Left \n');
          h = impoly;
          res = iptgetapi(h);
          l = res.getPosition();
        elseif strcmp(str,'r')
          fprintf(1,'Label Right \n');
          h = impoly;
          res = iptgetapi(h);
          r = res.getPosition();
        elseif strcmp(str,'t')
          fprintf(1,'Label Top \n');
          h = impoly;
          res = iptgetapi(h);
          t = res.getPosition();
        else
          fprintf(1,'illegal option\n');
        end
      end
    end
