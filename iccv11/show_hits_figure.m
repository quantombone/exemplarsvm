function show_hits_figure(models,topboxes,I,extraI,extraI2)
%Show a figure with the detections of the exemplar svm model
%Tomasz Malisiewicz(tomasz@cmu.edu)


topboxes = topboxes(1:min(5,size(topboxes,1)),:);
% if ~exist('score_masks','var')
%   for i = 1:size(topboxes,1)
%     score_masks{i}.scoremask = rand(20,20,1);
%     score_masks{i}.hogdet = zeros(20,20,1);
%     score_masks{i}.hogw = zeros(20,20,1);
%   end
% end

%show the parse in terms of exemplar associations

%use colors where 'hot' aka red means high score, and 'cold' aka
%blue means low score
colors = jet(size(topboxes,1));
colors = colors(end:-1:1,:);

subplot(2,3,1)
imagesc(I)
axis image
axis off
title('Input Image')
subplot(2,3,4)
PADDER = 100;
Ipad = pad_image(I,PADDER);
imagesc(I)
for q = size(topboxes,1):-1:1
  plot_bbox(topboxes(q,:),'',colors(q,:),colors(q,:))
end
axis image
axis off

title(sprintf('Box Cluster size %d',size(topboxes,1)))
exshows = cell(0,1);

for q = size(topboxes,1):-1:1
  if q>size(topboxes,1)
    continue
  end

  bb = topboxes(q,1:4);

  %titler=sprintf('E=%d S=%.3f',topboxes(q,6),topboxes(q,end));
  %plot_bbox(bb+PADDER);%,titler);
  %plot_bbox(bb);
  bb = bb + PADDER;
  r1 = [round(bb(2)) round(bb(4))];
  r2 = [round(bb(1)) round(bb(3))];
  r1 = cap_range(r1,1,size(Ipad,1));
  r2 = cap_range(r2,1,size(Ipad,2));

  mask = zeros(size(Ipad,1),size(Ipad,2));
  mask(r1(1):r1(2), r2(1):r2(2))=1;
  mask = logical(mask);
  chunks{q} = maskCropper(Ipad, mask);
  exshows{q} = [];
  %counter = counter+2;
end

%title(sprintf('Testing Images %s',curid))

%sss = cellfun2(@(x)x.scoremask(:),score_masks);
%sss = cat(1,sss{:});
%bnd_a = min(sss)-.00001;
%bnd_b = max(sss)+.00001;


N = min(5,size(topboxes,1));
for i = 1:N


  mid = topboxes(i,6);
  % subplot(N,15,15*i-3-5)
  % imagesc(score_masks{i}.hogdet);
  % title('HOG@window');
  % axis image
  % axis off
  
  % subplot(N,15,15*i-4-5)
  % imagesc(score_masks{i}.hogw);
  % title('W');
  % axis image
  % axis off
  

  subplot(N,6,6*(i-1)+3)
  imagesc(chunks{i});
  plot_bbox([1 1 size(chunks{i},2) size(chunks{i},1)],'',colors(i,:),colors(i,:));
  title([num2str(topboxes(i,end))]);% ' ' models{topboxes(i,6)}.models_name]);
  axis image
  axis off
  
  % subplot(N,4,4*(i-1)+4)
  % imagesc(score_masks{i}.scoremask,[bnd_a bnd_b]);  
  % colormap jet
  % title('Score Mask');
  % axis image
  % axis off
  
  subplot(N,6,6*(i-1)+4)  
  Iex = get_exemplar_icon(models,mid);
  if topboxes(i,7) == 1
    Iex = flip_image(Iex);
  end
  imagesc(Iex)
  plot_bbox([1 1 size(Iex,2) size(Iex,1)],'',colors(i,:),colors(i,:));
  axis image
  axis off
  %q = 13;

  
  % if ~isfield(models{mid},'subI') | length(models{mid}.subI)==0

  %   if isfield(models{mid},'I')
  %     Iex = models{mid}.I;
  %   else
  %     %if image is not present, then we have a VOC image, which we
  %     %load from the VOC path
      
  %   end
    

  %   if 0
  %   hold on;      
  %   plot_bbox(models{mid}.coarse_box(q,:),'',[0 1 0],[0 1 0]);

  %   hold on;
  %   plot_bbox(models{mid}.gt_box,'',[1 0 0],[1 0 0]);
  %   end  
  %   %Iex = zeros(10,10,3);
  %   %imagesc(Iex)
  % else
  %   Iex = models{mid}.subI;
  %   imagesc(Iex)
  %   axis image
  %   axis off
  % end

  id_string = '';
  if isfield(models{abs(mid)},'curid')
    try
      oid = models{abs(mid)}.objectid;
    catch
      oid = -1;
    end
    id_string = sprintf('%s.%d',models{abs(mid)}.curid, ...
                        oid);
  else
    %id_string = sprintf('dalal@%s',models{mid}.cls);
    id_string = sprintf('%s@%s',models{abs(mid)}.models_name,models{abs(mid)}.cls);
  end
  lrstring='';
  if topboxes(i,7) == 1
    lrstring = '.LR';
  end
  
  title(sprintf('E:%s%s',id_string,lrstring),'interpreter','none');
  
  
  %subplot(N,3,3*i);
  %exemplar_inpaint(I, topboxes(i,:), models{mid});
  
  %axis image
  %axis off
  
end

if exist('extraI','var')
  subplot(2,3,3)
  imagesc(extraI)
  axis image
  axis off
  title('Appearance Transfer')
end

if exist('extraI2','var')
  subplot(2,3,6)
  imagesc(extraI2)
  axis image
  axis off
  title('Mean App Transfer')
end