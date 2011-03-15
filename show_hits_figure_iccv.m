function NR=show_hits_figure_iccv(models,topboxes,I,extraI,exemplar_overlay,gtim)
%Show a figure with the detections of the exemplar svm model
%Tomasz Malisiewicz(tomasz@cmu.edu)
VOCinit;

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

add_one = 1;

% if isfield(exemplar_overlay,'friendbb') & ...
%       size(exemplar_overlay.friendbb,1)>0
%   %ha = tight_subplot(2, 2, 0, .1, .1);
%   add_one = 1;
  
% elseif isfield(exemplar_overlay,'overlay2')
%   add_one = 1;
% end
  
figure(1)
clf
if add_one == 1
  ha = tight_subplot(2, 2, .05, ...
                     .1, .01);
  % hb = tight_subplot(4, 2, .05, ...
  %                    .1, .01);
  
  % for q = 1:8
  %   axes(hb(q));
  %   imagesc(ones(100,100,3));
  %   axis image
  %   axis off
  % end

  % keyboard
  NR = [2 2];
else
  ha = tight_subplot(1, 3, .05, ...
                     .1, .01);
  NR = [3 1];
end

PADDER = 100;
Ipad = pad_image(I,PADDER);
%imagesc(I)
% for q = 1%size(topboxes,1):-1:1
%   plot_bbox(topboxes(q,:),'',[1 0 0],[1 0 0]);%colors(q,:))
% end
%axis image
%axis off
%title('Input Image')
%title(sprintf('%s Detection Window',models{topboxes(q,6)}.cls))
%title(sprintf('Box Cluster size %d',size(topboxes,1)))
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


N = min(1,size(topboxes,1));
for i = 1:N

  mid = topboxes(i,6);

  hogpic = HOGpicture(models{topboxes(i,6)}.model.w);
  if isfield(models{topboxes(i,6)},'FLIP_LR')
    hogpic = flip_image(hogpic);
  end
  
  mid
  Iex = get_exemplar_icon(models, mid);


  Iex1 = imresize(chunks{i},[size(Iex,1) size(Iex,2)]);
  Iex1 = max(0.0,min(1.0,Iex1));
  PPP = round(size(Iex,1)*.3);
  Iex2 = Iex1(1:PPP,:,:)*0+1;
  Ishow = cat(1,Iex1,Iex2,Iex);
  bb1 = [1 1 size(Iex,2) size(Iex,1)];
  bb2 = bb1;
  bb2([2 4]) = bb2([2 4]) + size(Iex,1)+PPP;
  
  axes(ha(2));
  imagesc(Ishow)

  axis image
  axis off

  plot_bbox(bb1,'',[1 1 0],[1 1 0],0,[2 1])
  plot_bbox(bb2,'Exemplar',[0 1 0],[0 1 0],0,[2 1])
  
  title('Association')

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
    id_string = sprintf('%s@%s',models{abs(mid)}.models_name,models{abs(mid)}.cls);
  end
  lrstring='';
  if isfield(models{abs(mid)},'FLIP_LR')
    lrstring = '.LR';
  end

  sprintf('Exemplar: %s%s',id_string,lrstring)%,'interpreter','none');
    
end

axes(ha(1));
imagesc(I)
axis image
axis off
title('Input Image')

if exist('extraI','var')
  %ha = tight_subplot(2, 2, 0, .1, .1);
  axes(ha(3));

  %subplot(2,2,3)
  imagesc(extraI)
  clipped_top = clip_to_image(topboxes(i,:),[1 1 size(extraI,2) ...
                    size(extraI,1)]);
  clipped_top([1 2]) = clipped_top([1 2])+2;
  clipped_top([3 4]) = clipped_top([3 4])-2;
  plot_bbox(clipped_top,'E',[0 1 0],[0 1 0],0,[2 1]);
  axis image
  axis off
  title('Exemplar Inpainting')
  
  if add_one == 1
  %transfer objects here
  
  is_complement = sum(ismember(models{1}.cls,{'bicycle', ...
                    'motorbike','horse'}));
  
  if isfield(exemplar_overlay,'friendbb') & ...
        size(exemplar_overlay.friendbb,1)>0

    axes(ha(4));
  

    imagesc(I)
    axis image
    axis off
    

    c = jet(20);

    for q = 1:size(exemplar_overlay.friendbb,1)
      [aa,bb] = find(ismember(VOCopts.classes, ...
                              exemplar_overlay.friendclass{q}));
      curcolor = c(aa,:);
      %Yellow color always
      %curcolor = [1 1 0];
      curbb = exemplar_overlay.friendbb(q,:);      
      %curbb(1:2) = curbb(1:2)-1;
      %curbb(3:4) = curbb(3:4)+1;
      curbb = clip_to_image(curbb,[1 1 size(I,2) size(I,1)]);

      plot_bbox(curbb,...
                exemplar_overlay.friendclass{q},curcolor,curcolor,1,[2 ...
                    1]);

      [aa,bb] = find(ismember(VOCopts.classes, ...
                              models{1}.cls));
      curcolor = c(aa,:);
      curcolor = [1 1 0];

      curbb = clip_to_image(topboxes(1,:),[1 1 size(I,2) size(I, ...
                                                  1)]);

      plot_bbox(curbb,models{1}.cls,curcolor,curcolor,0,[2 1]);
      title(sprintf('Detection: Predicted %s', ...
                    exemplar_overlay.friendclass{1}));
      

    end
  elseif ~is_complement && isfield(exemplar_overlay,'overlay2')
    
    axes(ha(4));
    imagesc(exemplar_overlay ...
            .overlay2.segI);
    title(exemplar_overlay.overlay2.title);
    axis image
    axis off
    
    if isfield(exemplar_overlay.overlay2,'faces')
      faces = exemplar_overlay.overlay2.faces;
      for q = 1:length(faces)
        if length(faces{q})>0
          f = faces{q};
          %%estimate Xform here:
          source = topboxes(1,:);
          target = models{source(6)}.gt_box;
          
          if isfield(models{source(6)},'FLIP_LR')
            target = flip_box(target,models{source(6)}.sizeI);
            f = flip_faces(f,models{source(6)}.sizeI);
            
            fprintf(1,'got here for a flip')
            %source = flip_box(source,size(I));
          end
          clear xs ys
          xs(:,1) = target([1 2])';
          xs(:,2) = target([3 2])';
          xs(:,3) = target([1 4])';
          xs(:,4) = target([3 4])';
          
          ys(:,1) = source([1 2])';
          ys(:,2) = source([3 2])';
          ys(:,3) = source([1 4])';
          ys(:,4) = source([3 4])';
          
          xs(3,:) = 1;
          ys(3,:) = 1;

          A=ys*pinv(xs);
          A(abs(A)<.000001) = 0;
          
          f2 = f';
          f2(3,:) = 1;
          
          f2 = A*f2;
          f2(3,:) = [];
          f = f2';
          
          hold on;
          plot([f(:,1); f(1,1)],[f(:,2); f(1,2)],'k','LineWidth', ...
               4);

        end
      end
    end
    
  else
    axes(ha(4))
    imagesc(I)
    axis image
    axis off
    
    titler = sprintf('Detection');
    if isfield(models{mid},'friendclass')
      titler = [titler sprintf(': No %s',models{mid}.friendclass)];
    end
    title(titler);

    [aa,bb] = find(ismember(VOCopts.classes, ...
                            models{1}.cls));
    c = jet(20);
    curcolor = c(aa,:);
    curcolor = [1 1 0];
    

    clipped_top = clip_to_image(topboxes(1,:),[1 1 size(extraI,2) ...
                    size(extraI,1)]);
    clipped_top([1 2]) = clipped_top([1 2])+2;
    clipped_top([3 4]) = clipped_top([3 4])-2;
    plot_bbox(clipped_top,models{1}.cls,curcolor,curcolor,0,[2 1]);
  end
end  

end
