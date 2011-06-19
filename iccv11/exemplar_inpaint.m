function exemplar_overlay = exemplar_inpaint(detection_box, model, ...
                                             stuff)
%paint the exemplar into the image I 

%% show gt bb with os in [.5, 1] as correct
%% show gt bb with os in [MAXOS_THRESH,.5) as incorrect
%% do not even show gt bb if os is in [0,MAXOS_THRESH)
MAXOS_THRESH = .2;

SHOW_TITLES = 0;

%if enabled, show top friend
SHOW_FRIENDS = 0;

%if enabled, steal segmentation
STEAL_SEGMENTATION = 0;

%if enabled, steal 3D
STEAL_3D = 0;

I = stuff.I;

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
xform_cd = find_xform(g, detection_box(1:4));

% if STEAL_3D
%   extra_3d = load_3d_stuff(model);
% else
%   extra_3d = [];
% end

loadseg = 0;
[mini_overlay.I, mini_overlay.alphamask] = ...
    get_exemplar_icon({model},1,0,1,loadseg,stuff.dataset_params);

%[mini_overlay.I, mini_overlay.alphamask] = ...
%    get_seg_icon({model},1,stuff.dataset_params);

[exemplar_overlay.I, exemplar_overlay.alphamask] = ...
    insert_exemplar(I, mini_overlay, detection_box);

% if strcmp(model.cls,'bus')
%   [Iex2,alphamask2,faces] = get_geometry_icon({model},1);

%   [exemplar_overlay2,Iex2] = ...
%       load_exemplar_overlay(model, Iex2, alphamask2, detection_box(7));

%   [exemplar_overlay2.segI,exemplar_overlay2.mask] = ...
%       insert_exemplar(I,exemplar_overlay2, ...
%                       xform_cd);
  
%   exemplar_overlay2.Iex = Iex2;
%   exemplar_overlay2.title = 'Qualitative Geometry';
%   exemplar_overlay2.faces = faces;
%   exemplar_overlay.overlay2 = exemplar_overlay2;  
  
% else
  
%   [Iexs,alphamasks] = get_seg_icon({model},1,stuff.dataset_params);
%   if length(Iexs) > 0
%     [exemplar_overlay2] = ...
%         load_exemplar_overlay(model, Iexs, alphamasks,detection_box(7));
    
%     [exemplar_overlay2.segI,exemplar_overlay2.mask] = ...
%         insert_exemplar(I,exemplar_overlay2, ...
%                         xform_cd);
%     exemplar_overlay2.Iex = Iexs;
%     exemplar_overlay2.title = 'Segmentation';
%     exemplar_overlay.overlay2 = exemplar_overlay2;  
%   end
% end


if SHOW_FRIENDS == 1
  fprintf(1,'transferring friends\n');
  dbox = detection_box;
  dbox(:,6) = 1;
  [exemplar_overlay.friendbb, exemplar_overlay.friendclass] = ...
      transfer_friends({model},{dbox});
  exemplar_overlay.friendbb = exemplar_overlay.friendbb{1};
  exemplar_overlay.friendclass = exemplar_overlay.friendclass{1};
end

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
  
else
  return;
end

extra_3d.box = exemplar_frame;
extra_3d.I = subI;
extra_3d.alphamask = submask;

function [I2,mask] = insert_exemplar(I,overlay,d)
%Given a target image, an overlay, and the transform, place it
%inside the new image


if d(7) == 1
  overlay.I = flip_image(overlay.I);
  overlay.alphamask = flip_image(overlay.alphamask);
end
target = round(d);

newsize = [target(4)-target(2)+1 target(3)- ...
                    target(1)+1];
newI = imresize(overlay.I,newsize,'nearest');
newalpha = imresize(overlay.alphamask, newsize,'nearest');
I2 = I;

newalpha = repmat(newalpha,[1 1 3]);

PAD = round(size(I2,1)*.8);
I2 = pad_image(I2,PAD);

mask = zeros(size(I2,1),size(I2,2));
%%% HACK making black images
%I2 = I2*0;

target = target+PAD;

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

I2 = pad_image(I2,-PAD);
I2 = max(0.0,min(1.0,I2));

mask = pad_image(mask,-PAD);
mask = max(0.0,min(1.0,mask));

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
