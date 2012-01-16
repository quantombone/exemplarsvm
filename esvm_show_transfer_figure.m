function NR = esvm_show_transfer_figure(I, models, topboxes, overlays, ...
                                        current_rank, corr)
%Show a figure with the detections of the exemplar svm model
%NOTE(TJM): this function needs cleanup
%Tomasz Malisiewicz(tomasz@cmu.edu)

topboxes = topboxes(1:min(2,size(topboxes,1)),:);

%if enabled, try to transfer objects too
%add_one = 1;
add_one = 0;
  
figure(1)
clf
if add_one == 1
  ha = tight_subplot(2, 2, .05, ...
                     .1, .01);
  NR = [2 2];
else
  ha = tight_subplot(1, 2, .05, ...
                     .01, .01);
  %hb = tight_subplot(1, 2, .05, ...
  %                   .1, .01);
  NR = [2 2];
  
end

PADDER = 100;
Ipad = pad_image(I,PADDER);

%Cut out detection chunks
for q = size(topboxes,1):-1:1
  if q>size(topboxes,1)
    continue
  end

  bb = topboxes(q,1:4);

  bb = bb + PADDER;
  r1 = [round(bb(2)) round(bb(4))];
  r2 = [round(bb(1)) round(bb(3))];
  r1 = cap_range(r1,1,size(Ipad,1));
  r2 = cap_range(r2,1,size(Ipad,2));

  mask = zeros(size(Ipad,1),size(Ipad,2));
  mask(r1(1):r1(2), r2(1):r2(2))=1;
  mask = logical(mask);
  chunks{q} = maskCropper(Ipad, mask);
end

%%NOTE: below is a loop, but we only process the first element
%since we only show the top association
N = min(1,size(topboxes,1));
for i = 1:N

  mid = topboxes(i,6);
  flip = topboxes(i,7);
  hogpic = HOGpicture(models{topboxes(i,6)}.model.w);
  hogpic = jettify(hogpic);
  if flip == 1
    hogpic = flip_image(hogpic);
  end
  
  Iex = get_exemplar_icon(models, mid, topboxes(i,7));
  
  Iex1 = imresize(chunks{i},[size(Iex,1) size(Iex,2)]);
  Iex1 = max(0.0,min(1.0,Iex1));
  
  Iex1 = imresize(hogpic,[size(Iex1,1) size(Iex,2)], 'nearest');
  Iex1 = max(0.0,min(1.0,Iex1));
  
  PPP = round(size(Iex,1)*.2);
  Iex2 = Iex1(1:PPP,:,:)*0+1;
  Ishow = cat(1,Iex1,Iex2,Iex);

  
  SHOW_SEG = 1;
  if SHOW_SEG == 1
    
    Ia = (overlays{1}.mini_overlay.I.* ...
              repmat(overlays{1}.mini_overlay.alphamask,[1 1 3]));
    Ib = Iex;
    alphamap = overlays{1}.mini_overlay.alphamask;
    alphamap(alphamap>0) = .5;
    alphamap(alphamap==0) = .7;
    alphamap = repmat(alphamap,[1 1 3]);
    Itotal = Ia.*alphamap + Ib.*(1-alphamap);
    
    Imeta = Itotal;
    offset = [0 size(Ishow,1)+size(Iex2,1)];
 
    %Ishow = cat(1,Ishow,Iex2,Imeta);
  
  elseif strcmp(models{mid}.cls,'bus')
    m1 = repmat(double(overlays{1}.mini_overlay.alphamask>0)*.8,[1 ...
                    1 3]);
    m2 = repmat(double(overlays{1}.mini_overlay.alphamask>0)*.2,[1 ...
                    1 3]);
    m2(m2(:)==0) = 1;

    Imeta = overlays{1}.mini_overlay.I.*m1 +...
            Iex.*m2;
    offset = [0 size(Ishow,1)+size(Iex2,1)]
    overlays{1}.mini_overlay.faces = ...
        cellfun2(@ ...
                 (x)bsxfun(@plus,x,offset),overlays{1}.mini_overlay.faces);
 
    %Ishow = cat(1,Ishow,Iex2,Imeta);
  end
 
  bb1 = [1 1 size(Iex,2) size(Iex,1)];
  bb2 = bb1;
  
  bb2([2 4]) = bb2([2 4]) + size(Iex,1) + PPP;
  axes(ha(1));
  imagesc(Ishow)

  axis image
  axis off

  bouter = [1 1 size(Ishow,2) size(Ishow,1)];
  bouter(1) = bouter(1) - round(size(Ishow,2)*.05);
  bouter(2) = bouter(2) - round(size(Ishow,1)*.05);
  bouter(3) = bouter(3) + round(size(Ishow,2)*.05);
  bouter(4) = bouter(4) + round(size(Ishow,1)*.05);
  plot_bbox(bouter, '', [0 0 0], [0 0 0], 0, [2 1])

  if isfield(overlays{1}.mini_overlay,'faces')
    plot_faces(overlays{1}.mini_overlay.faces);
  end
  
  axis image
  curcolor = [0 1 0];
  if corr == 0
    curcolor = [1 0 0 ];
  end

  plot_bbox(bb1, sprintf('Exemplar-SVM %s %d',models{mid}.cls,mid), ...
            curcolor, curcolor, 0, [2 1])
  
  %sprintf('Exemplar.%d',mid)
  
  plot_bbox(bb2, sprintf('Exemplar Image %d',mid), [1 1 0], [1 1 0], 0, [2 1])
  
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
  if flip == 1
    lrstring = '.LR';
  end

  sprintf('Exemplar: %s%s',id_string,lrstring);
    
end

extraI = overlays{1}.I;

clipped_top = clip_to_image(topboxes(1,:),[1 1 size(extraI,2) ...
                    size(extraI,1)]);
clipped_top([1 2]) = clipped_top([1 2])+2;
clipped_top([3 4]) = clipped_top([3 4])-2;

axes(ha(2));

PPP = round(size(I,1)*.05);
I2 = I(1:PPP,:,:)*0+1;
Ishow = cat(1,I,I2,extraI);
%Ishow = extraI;

imagesc(Ishow)
axis image
axis off
%title(sprintf('Detection (Rank=%d, Score=%.3f)',current_rank, ...
%              topboxes(1,end)))


curcolor = [0 1 0];
if corr == 0
  curcolor = [1 0 0 ];
end

plot_bbox(clipped_top,sprintf('%s: %.3f',models{1}.cls,clipped_top(end)),...
          curcolor, curcolor,0,[2 1]);

if 0
  %%NOTE: not sure why this is not working?
%% find xform from exemplar onto new detection
start_box = [1 1 size(Iex,2) size(Iex,1)];
end_box = topboxes(1,1:4);
xform = find_xform(start_box,end_box);

faces2 = cellfun2(@(x)apply_xform(x,xform), ...
                  overlays{1}.mini_overlay.faces);

offset = [0 size(I,1)+size(I2,1)];
offset = [0 0];
faces2 = cellfun2(@ ...
                  (x)bsxfun(@plus,x,offset),faces2);
end
%plot_faces(faces2);

%axes(ha(3));
%imagesc(extraI);

%sprintf('Exemplar.%d',mid)
%orig green
clipped_top([2 4]) = clipped_top([2 4]) + size(I,1)+size(I2,1);

%dont do yellow box for segmentation
if 1 %%~strcmp(models{mid}.cls,'bus')
  plot_bbox(clipped_top,'Exemplar',[1 1 0],[1 1 0],0,[2 1]);
end
axis image
axis off
%title('Appearance Transfer')%sprintf('Exemplar Inpainting: %.3f',topboxes(1,end)))
%title(sprintf('Exemplar Inpainting: %.3f',topboxes(1,end)))

title('Exemplar-SVM detection','FontSize',12);

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
      curbb = exemplar_overlay.friendbb(q,:);      
      curbb = clip_to_image(curbb,[1 1 size(I,2) size(I,1)]);

      plot_bbox(curbb,...
                exemplar_overlay.friendclass{q},curcolor,curcolor,1,[2 ...
                    1]);

      [aa,bb] = find(ismember(VOCopts.classes, ...
                              models{1}.cls));
      curcolor = c(aa,:);
      curcolor = [1 1 0];
      curcolor = [0 1 0];
      if corr == 0
        curcolor = [1 0 0 ];
      end

      curbb = clip_to_image(topboxes(1,:),[1 1 size(I,2) size(I, ...
                                                  1)]);

      plot_bbox(curbb,models{1}.cls,curcolor,curcolor,0,[2 1]);
      title(sprintf('Detection: Predicted %s', ...
                    exemplar_overlay.friendclass{1}));
      

    end
  elseif ~is_complement && isfield(exemplar_overlay,'overlay2')
    %% SHOW TRANSFERRED BUSES
    axes(ha(4));
    imagesc(exemplar_overlay ...
            .overlay2.segI);
    title(exemplar_overlay.overlay2.title);
    axis image
    axis off

    %Faces are for transferring geometry
    if isfield(exemplar_overlay.overlay2,'faces')
      faces = exemplar_overlay.overlay2.faces;
      for q = 1:length(faces)
        if length(faces{q})>0
          f = faces{q};
          %%estimate Xform here:
          source = topboxes(1,:);
          target = models{source(6)}.gt_box;
          
          if topboxes(1,7) == 1 
            target = flip_box(target,models{source(6)}.sizeI);
            f = flip_faces(f,models{source(6)}.sizeI);
            
            fprintf(1,'got here for a flip')
            %source = flip_box(source,size(I));
          end
          
          %%BUG: we should be using the new find_xform function...
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
    %% SHOW FRIENDS
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


