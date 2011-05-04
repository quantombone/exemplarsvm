function exemplar_overlay = exemplar_inpaint(detection_box, model, stuff)
%paint the exemplar into the image I 

%% show gt bb with os in [.5,1] as correct
%% show gt bb with os in [MAXOS_THRESH,.5) as incorrect
%% do not even show gt bb if os is in [0,MAXOS_THRESH)
MAXOS_THRESH = .2;

SHOW_TITLES = 0;

%if enabled, show top friend`
SHOW_FRIENDS = 1;

%if enabled, steal segmentation
STEAL_SEGMENTATION = 0;

%if enabled, steal 3D
STEAL_3D = 0;

VOCinit;


curid = model.curid;
I = im2double(imread(sprintf(VOCopts.imgpath, curid)));

if 0
recs = PASreadrecord(sprintf(VOCopts.annopath, curid));
gtboxes = cat(1, recs.objects.bbox);
gtclasses = {recs.objects.class};
validboxes = find(ismember(gtclasses,model.cls));
gtboxes = gtboxes(validboxes,:);

os = getosmatrix_bb(detection_box, gtboxes);
[maxos,gtid] = max(os);
gt_box = gtboxes(gtid,:);

NEGPAD = 20;
border_box = [1-NEGPAD 1-NEGPAD size(I,2)+NEGPAD size(I,1)+NEGPAD];
border_color = [1 0 0];

%NEGPAD2 = 25;
%border_box2 = [1-NEGPAD2 1-NEGPAD2 size(I,2)+NEGPAD2 size(I,1)+NEGPAD2];

is_correct = 0;
if maxos > .5
  gt_color = [0 1 0];
  is_correct = 1;
elseif maxos > MAXOS_THRESH
  gt_color = [1 0 0];
end
end

if 0 %strcmp(model.cls,'bus')
  [Iex,alphamask] = get_geometry_icon({model},1);
else
  %[Iex,alphamask] = get_exemplar_icon({model},1);
  Iex = im2double(imread(sprintf(VOCopts.imgpath,model.curid)));
  alphamask = ones(size(Iex,1),size(Iex,2),1);
end


if 0
  if length(Iex) == 0
    [Iex,alphamask] = get_seg_icon({model},1);
    if length(Iex) == 0
      Iex = im2double(imread(sprintf(VOCopts.imgpath,model.curid)));
      alphamask = ones(size(Iex,1),size(Iex,2),1);
    end
  end
end
  


if 0 %strcmp(model.cls,'motorbike')
  resdir = '/nfs/baikal/tmalisie/iccv11-gcsegs/';
  resfile = sprintf('%s/%s.%05d.mat',resdir,model.curid, ...
                    model.objectid);

  try
  res = load(resfile);
  %alphamask = double(res.mask==1);
  seg2 = res.mask;
  %seg2 = repmat(res.mask,[1 1 3]);
  %seg2 = faces2colors(res.res.seg);
  Iex = repmat(seg2,[1 1 3]);
  
  catch
    fprintf(1,'error here\n');
  end
end

d = detection_box(1:4);

if SHOW_FRIENDS == 1
  %get c's friends
  
  if isfield(model,'friendclass')
    model = initialize_with_friends(model,model.friendclass);
  end
  
  if sum(ismember(model.cls,{'bicycle','motorbike','horse'}))==0
    SHOW_FRIENDS = 0;
  end
else
  friends = [];
end

%% get ground truth and its projection
g = model.gt_box;

%%% estimate (translation,scale) from exemplar image to test image
xform_cd = find_xform(g, d);

% if STEAL_3D
%   extra_3d = load_3d_stuff(model);
% else
%   extra_3d = [];
% end

[exemplar_overlay] = ...
    load_exemplar_overlay(model, Iex, alphamask);

[exemplar_overlay.segI,exemplar_overlay.mask] = ...
    insert_exemplar(I,exemplar_overlay, ...
                                        xform_cd);

if strcmp(model.cls,'bus')
  [Iex2,alphamask2,faces] = get_geometry_icon({model},1);
  [exemplar_overlay2] = ...
      load_exemplar_overlay(model, Iex2, alphamask2);
  
  [exemplar_overlay2.segI,exemplar_overlay2.mask] = ...
      insert_exemplar(I,exemplar_overlay2, ...
                      xform_cd);
  exemplar_overlay2.Iex = Iex2;
  exemplar_overlay2.title = 'Qualitative Geometry';
  exemplar_overlay2.faces = faces;
  exemplar_overlay.overlay2 = exemplar_overlay2;  
  
else
  
  [Iexs,alphamasks] = get_seg_icon({model},1);
  if length(Iexs) > 0
    [exemplar_overlay2] = ...
        load_exemplar_overlay(model, Iexs, alphamasks);
    
    [exemplar_overlay2.segI,exemplar_overlay2.mask] = ...
        insert_exemplar(I,exemplar_overlay2, ...
                        xform_cd);
    exemplar_overlay2.Iex = Iexs;
    exemplar_overlay2.title = 'Segmentation';
    exemplar_overlay.overlay2 = exemplar_overlay2;  
  end
  
end



exemplar_overlay.Iex = Iex;

dbox = detection_box;
dbox(:,6) = 1;

if SHOW_FRIENDS == 1
  fprintf(1,'transferring friends\n');
  [exemplar_overlay.friendbb,exemplar_overlay.friendclass] = transfer_friends({model},{dbox});
  exemplar_overlay.friendbb = exemplar_overlay.friendbb{1};
  exemplar_overlay.friendclass = exemplar_overlay.friendclass{1};
end

%exemplar_overlay.friendbb = friendbb;
%exemplar_overlay.friendclass = friendclass;


return;

figure(1)
clf
ha = tight_subplot(1, 1, 0, .1, .1);
axes(ha(1));

%% Inside first tile we show the ground truth projection
%axes(ha(1));
imagesc(I)

if SHOW_TITLES
  title(sprintf('Detection %d: Test Image %s, OS=%.3f',...
                stuff.rank,curid,maxos));
end

if ~STEAL_SEGMENTATION

  if ~STEAL_3D
    if maxos > MAXOS_THRESH
      plot_bbox(gt_box, '', gt_color, gt_color, 1);
    end
    
    if ~is_correct
      plot_bbox(border_box, '', border_color, border_color, 0);
    end
    
    %always drop white border
    %plot_bbox(border_box2, '', [0 0 0],[0 0 0], 0);
  end
  dcolor1 = [0 0 1];
  dcolor2 = [.2 1 .2];
  current_label = model.cls;
  if STEAL_3D
    dcolor1 = [0 1 0];
    dcolor2 = [0 1 0];
    current_label = '';
  end
  
  %plot_bbox(gprime, current_label, dcolor1, dcolor2);
  plot_bbox(d, current_label, dcolor1, dcolor2);
end

axis image
axis off
%pause;
%return;
save_me_as_pdf(stuff,1);

figure(2)
clf
ha = tight_subplot(1, 1, 0, .1, .1);
axes(ha(1))

titler = 'Exemplar Overlay';
if has_seg == 1
  titler = 'Exemplar/Seg Overlay';
end

if length(extra_3d) > 0
  %% Inside the second tile we show the 3D transfer
  imagesc(insert_exemplar(I,extra_3d,xform_cd))
  titler = '3D overlay';
else
  imagesc(insert_exemplar(I,exemplar_overlay,xform_cd))
end

if length(friends) > 0
  titler = [titler sprintf(' + ''%s'' Transfer',friendclass{1})];
end

if SHOW_TITLES
  title(titler);
end

for iii = 1:size(friendbb,1)
  plot_bbox(apply_xform(friendbb(iii,:),xform_cd),friendclass{iii});
end

if has_seg==0 && length(extra_3d)==0
  plot_bbox(gprime,model.cls, [0 0 1], [.2 1 .2])
end

%always drop white border
%plot_bbox(border_box2, '', [0 0 0],[0 0 0], 0);
%plot_bbox(border_box2, '', [1 1 1],[1 1 1], 0);

axis image
axis off

drawnow
save_me_as_pdf(stuff,2);

figure(3)
clf

%% Inside second tile we show the exemplar image
ha = tight_subplot(1, 1, 0, .1, .1);
axes(ha(1));
imagesc(Iex)

if SHOW_TITLES
  title(sprintf('Exemplar %s.%d', model.curid, model.objectid));
end

f1 = [0 0 1];
f2 = [1 0 0];
for iii = 1:length(friends)
  plot_bbox(friendbb(iii,:),friendclass{iii},f1,f2,1);
end
plot_bbox(g, model.cls, [0 0 1], [.2 1 .2],1)

axis image
axis off
save_me_as_pdf(stuff,3);

%%% if 3D, then show sliced exemplar and sliced 3D model

if STEAL_3D

  figure(1)
  clf
  ha = tight_subplot(2, 1, .1, .1, .1);
  axes(ha(1));
  imagesc(exemplar_overlay.I)
  axis image
  axis off
  title('Exemplar');
  %save_me_as_pdf(stuff,4);
  
  %figure(1)
  %clf
  %ha = tight_subplot(1, 1, 0, .1, .1);
  axes(ha(2));
  resI = pad_image(extra_3d.I,-20);
  resalpha = pad_image(extra_3d.alphamask,-20);
  resI(find(repmat(resalpha==0,[1 1 3])))=1;
  imagesc(resI)
  axis image
  axis off
  title('3D Model')
  save_me_as_pdf(stuff,5);
end

% function snapbox = snap_to_pixel_grid(box, cb)
% %% here we take the box in boxes which has non-integer locations
% %and snap it to a pixel grid such that it has the aspect ratio as
% %cb

% target_aspect = (cb(3)-cb(1)+1) / (cb(4)-cb(2)+1);
% newboxes = zeros(0,4);
% rough_box = round(box);
% ranges = -3:3;
% for a = ranges
%   for b = ranges
%     for c = ranges
%       for d = ranges
%         newboxes(end+1,:) = rough_box + [a b c d];
%       end
%     end
%   end
% end

% %get overlap scores
% curos = getosmatrix_bb(newboxes,box);
% aspects = (newboxes(:,3) - newboxes(:,1)+1) ./ (newboxes(:,4)- ...
%                                                 newboxes(:,2)+1);

% goods = find(aspects==target_aspect);
% [aa, bb] = max(curos(goods));

% snapbox = newboxes(goods(bb),:);

function extra_3d = load_3d_stuff(model)

%check if 3D stuff is present, and if it is load it
extra_3d = [];

extradir = '/nfs/hn22/tmalisie/ddip/renderings/';
filer = sprintf('%s/%s.%d*png', ...
                extradir,...
                model.curid, ...
                model.objectid);

fff = dir(filer);
if length(fff) > 0
  %% check if we have a 3D file present for this exemplar
  [Iex,map,Ialpha] = imread([extradir fff(1).name]);
  Iex = im2double(Iex);
  Ialpha = double(Ialpha)/255;
  %remove padding
  Ialpha = pad_image(Ialpha,-100);
  Iex = pad_image(Iex,-100);

  exemplar_frame = model.gt_box;
  exemplar_frame([1 2]) = exemplar_frame([1 2]) - 50;
  exemplar_frame([3 4]) = exemplar_frame([3 4]) + 50;
  
  subI = subarray(Iex,exemplar_frame(2),exemplar_frame(4),...
                  exemplar_frame(1),exemplar_frame(3),0);

  maskrep = repmat(Ialpha,[1 1 3]);
  submask = subarray(maskrep,exemplar_frame(2),exemplar_frame(4),...
                     exemplar_frame(1),exemplar_frame(3),0);
  submask = submask(:,:,1);
  %submask = Ialpha(exemplar_frame(2):exemplar_frame(4), ...
  %                 exemplar_frame(1):exemplar_frame(3), :);  

else
  return;
end

extra_3d.box = exemplar_frame;
extra_3d.I = subI;
extra_3d.alphamask = submask;

function [exemplar_overlay] = ...
    load_exemplar_overlay(model, Iex, alphamask)


%has_seg = 0;

g = model.gt_box;

subI = Iex(g(2):g(4), g(1):g(3), :);
alphamask = alphamask(g(2):g(4),g(1):g(3));

exemplar_overlay.box = g;
exemplar_overlay.I = subI;
exemplar_overlay.alphamask = alphamask;
%=v2e0h;exemplar_overlay.I = exemplar_overlay.I.*(repmat(alphamask,[1 1 3]));

if isfield(model,'FLIP_LR') && model.FLIP_LR == 1
  exemplar_overlay.I = flip_image(exemplar_overlay.I);
  exemplar_overlay.alphamask = ...
      flip_image(exemplar_overlay.alphamask);
end

function [I2,mask] = insert_exemplar(I,overlay,xform_cd)
%Given a target image, an overlay, and the transform, place it
%inside the new image

target = apply_xform(overlay.box, xform_cd);
target = round(target);

newsize = [target(4)-target(2)+1 target(3)- ...
                    target(1)+1];
newI = imresize(overlay.I,newsize,'nearest');
newalpha = imresize(overlay.alphamask,newsize,'nearest');
I2 = I;

newalpha = repmat(newalpha,[1 1 3]);

PAD = round(size(I2,1)*.8);
I2 = pad_image(I2,PAD);

mask = zeros(size(I2,1),size(I2,2));
%%% HACK making black images
%I2 = I2*0;

target = target+PAD;

%make image black as default
%I2 = I2*0;
try
  targetx = target(2):target(4);
  targety = target(1):target(3);
  goodsx = find(targetx>=1 & targetx<=size(I2,1));
  goodsy = find(targety>=1 & targety<=size(I2,2));
  
  targetx = targetx(goodsx);
  targety = targety(goodsy);
  
  I2(targetx,targety,:) = ...
      I2(targetx,targety,:).*(1-newalpha(goodsx,goodsy,:)) + ...
      newI(goodsx,goodsy,:).*(newalpha(goodsx,goodsy,:));
  
  mask(targetx,targety,:) = 1;
  
catch
  fprintf(1,'insert exemplar bug\n');
  keyboard
end
I2 = pad_image(I2,-PAD);
I2 = max(0.0,min(1.0,I2));


mask = pad_image(mask,-PAD);
mask = max(0.0,min(1.0,mask));

if 0

rawbox = [0 0 size(overlay.I,2)-1 size(overlay.I,1)-1];

%get the reverse xform
reverse_xform = find_xform(target, rawbox);

target([1 2]) = ceil(target([1 2]));
target([3 4]) = floor(target([3 4]));

I2 = I;
for i = target(2):target(4)
  for j = target(1):target(3)
    curval = [j i j i];
    newcoord = apply_xform(curval, reverse_xform);
    
  end
end


end

function save_me_as_pdf(stuff,index)
fprintf(1,'saving disabled temporarily\n');
pause
return;
stuff.filer = strrep(stuff.filer,'.eps',sprintf('-%d.eps',index));

set(gcf,'PaperPosition',[0 0 3 3]);
set(gcf,'PaperSize',[3 3]);

print(gcf,'-depsc2','-r600','-painters',stuff.filer);
filer2 = strrep(stuff.filer,'.eps','.pdf');
unix(sprintf(['ps2pdf -dPDFSETTINGS=/prepress -dEPSCrop ' ...
              '%s %s'],stuff.filer,filer2));
delete(stuff.filer);
fprintf(1,'Just Wrote %s\n',filer2);
    