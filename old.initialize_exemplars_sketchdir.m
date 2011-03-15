function initialize_exemplars_sketchdir(sketchdir)
% Initialize script which given a cell array of images, finds a
% region in each image (by user interaction)
% fg:       a cell array of images
% [models]: a cell array of models used to automatically select
%           input window
% models2:  resulting cell array of model files
% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

N_PER_FRAME = 1;
%results_directory = ...
%    sprintf('/projects/ddip/fgdata/exemplars/');

results_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

%fprintf(1,'Writing Exemplars to directory
%%s\n',results_directory);


if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

if exist('models','var') & length(models)>0
    ws = cellfun2(@(x)x.model.w,models);
    bs = cellfun2(@(x)x.model.b,models);
end

models2 = cell(0,1);

files = dir([sketchdir '*png']);
for i = 1:length(files)
  fg{i} = [sketchdir files(i).name];
end

for i = 1:min(50,length(fg))
  
  Ibase = convert_to_I(fg{i});
  Ibase = im2double(Ibase);

  for objectid = 1:N_PER_FRAME
    
    filer = sprintf('%s/%d-%d.mat',results_directory,i,objectid);
    if fileexists(filer)
      continue
    end

    if ~exist('bbox','var')
      figure(1)
      clf
      imagesc(Ibase)
      title(sprintf('enter box coords %d',objectid));
      
      % try to automatically segment out
      % outsiderow = squeeze(Ibase(1,1,:));
      % Irow = reshape(Ibase,[],3)';
      % ddd = distSqr_fast(Irow,outsiderow);
      % ddd = reshape(ddd',size(Ibase,1),size(Ibase,2));
      % seg = bwmorph((ddd>0),'dilate',2);
      % [u,v] = find(seg);
      % x(1) = min(v);
      % x(2) = max(v);
      % y(1) = min(u);
      % y(2) = max(u);
      
      [x,y] = ginput(2);  
      bbox = round([x(1) y(1) x(2) y(2)]);
      plot_bbox(bbox)      
    elseif exist('models','var') & length(models)>0
      %Find single top detection
      [rs] = localizemeHOG(Ibase,ws,bs,-2.00,1,10);
      bbox = extract_bbs_from_rs(rs,Ibase,models);
    end
    I = Ibase;

    
    % Get initial dimensions
    w = bbox(3)-bbox(1)+1;
    h = bbox(4)-bbox(2)+1;
    
    %get center of GT box (we will align centers later)
    wmid = round((bbox(3)+bbox(1)+1)/2);
    hmid = round((bbox(4)+bbox(2)+1)/2);
    
    w1 = w;
    h1 = h;
    aspect = h / w;
    area = h * w;    

    area = max(min(area, 5000), 3000);
    
    % pick dimensions
    w = sqrt(area/aspect);
    h = w*aspect;
    
    sbin = 8;
    hg_size = [round(h/sbin) round(w/sbin)];
    
    %make even sizes, makes for an easier job getting the lcm
    hg_size = max(1,round(hg_size/2))*2;
    
    %% find new bbox with the exact ratio and exact area
    newaspect = hg_size(1) / hg_size(2);

    %% needs to be a multiple of the least-common-multiple
    multer = lcm(hg_size(2),sbin);
    
    neww = multer:multer:3000;
    newh = newaspect*neww;
    
    goods = find(((round(newh) - newh)==0) & ((round(neww) - neww)==0));
                 
    neww = neww(goods);
    newh = newh(goods);
    
    olda = w1*h1;
    newa = newh.*neww;
    
    subs = find(neww >= w1 & newh >= h1);
    neww = neww(subs);
    newh = newh(subs);
    newa = newa(subs);
    
    %% We choose dimensions such that the GT box is enclosed WITHIN
    %the coarse_box, but we find the tightest box
    [aa,bb] = min(abs(newa - olda));
    newh = newh(bb);
    neww = neww(bb);
        
    up = round([hmid-newh/2 wmid-neww/2]);
    newb = [up(2) up(1) up(2)+neww-1 up(1)+newh-1];

    os = getosmatrix_bb(newb,bbox);
    fprintf(1,'OS of new region with GT region %.3f\n',os);

    
    
    if 0
      %Here we show the difference between the GT box and the
      %selected box
      figure(1)
      clf
      imagesc(I)
      axis image
      plot_bbox(bbox)
      hold on;
      plot_bbox(newb,'',[1 0 0],[1 0 0]);
      title(sprintf('%s.%d OS %.3f, cellsize = [%d x %d]\n',curid,objectid,os,hg_size(1),hg_size(2)));
    end
    
    %do padding in case detection is outside image border
    extras = 500;
    I2 = pad_image(I,extras);
    
    %% Save the coarse box before we pad it
    coarse_box = newb;
    newb = newb+extras;
    
  
    sbin2 = (newb(3)-newb(1)+1) / hg_size(2);
  
    curx = (newb(2)-sbin2):(newb(4)+sbin2);
    cury = (newb(1)-sbin2):(newb(3)+sbin2);
    padI = zeros(length(curx),length(cury),3);
    goodx = find(curx>=1 & curx<=size(I2,1));
    goody = find(cury>=1 & cury<=size(I2,2));

    padI(goodx,goody,:) = I2(curx(goodx),cury(goody),:);

    
    %clear the model
    clear m model

    % here we save 25 little wiggles around the ground truth region
    model.x = zeros(prod(hg_size)*31,0);
    m.coarse_box = zeros(0,4);
    for cx = -2:2
      for cy = -2:2
        f = features(circshift(padI,[cx cy]),sbin2);
        cb = coarse_box;
        cb([1 3]) = cb([1 3]) + cy;
        cb([2 4]) = cb([2 4]) + cx;
        m.coarse_box(end+1,:) = cb; 
        model.x(:,end+1) = f(:);
      end
    end
    

 
    model.hg_size = [hg_size 31];
    model.w = reshape(mean(model.x,2),model.hg_size);
    model.b = 0;
    model.t = 'initial_model';
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    model.mine_ids = [];
    
    m.gt_box = bbox;
    m.model = model;
    m.index = i;
    m.I = I;
    m.cls = 'fg';
    
    % %taking center feature is better
    % x = reshape(model.x(:,13),hg_size(1),hg_size(2),[]);
    % %x = reshape(mean(model.x,2),hg_size(1),hg_size(2),[]);
    % x = hog_normalize(x);
    
    % clear bbb;
    % bbb = cell(0,1);
    
    % % subI = pad_image(I,500);
    % % P = 100;
    % % subI = subI(round(m.gt_box(2)-P+500):round(m.gt_box(4)+P+500), ...
    % %             round(m.gt_box(1)-P+500):round(m.gt_box(3)+P+500),:);
    
    % clf
    % imagesc(I)

    % xxx = cell(0,1);
    % for cx = -3:3:3
    %   for cy = -3:3:3
    %     I2 = circshift2(I,[cx cy]);
    %     [rs,t] = localizemeHOGnormalized(I2,{x},{0},-100,20,50);
    %     xxx{end+1} = rs.support_grid{1};
    %     rs.support_grid = [];
    %     mmm{1}.model.w = x;
    %     mmm{1}.model.hg_size = size(mmm{1}.model.w);
    %     mmm{1}.model.hg_size = mmm{1}.model.hg_size(1:2);
    %     mmm{1}.model.b = 0;
    %     mmm{1}.cls = 'fg';
    %     [bbs] = extract_bbs_from_rs(rs,I,mmm);
    %     bbs(:,[1 3]) = bbs(:,[1 3])+cy;
    %     bbs(:,[2 4]) = bbs(:,[2 4])+cx;
    %     bbb{end+1} = bbs;
    %   end
    % end
    % xxx = cat(2,xxx{:});
    % bbb = cat(1,bbb{:});
    % [aa,bb] = sort(bbb(:,end),'descend');
    % bbb = bbb(bb,:);
    
    % %bbs(:,1:4) = bbs(:,1:4)/2.0;
    % for i = 1:5, plot_bbox(bbb(i,:)); end
    % for i = 1:25, plot_bbox(m.coarse_box(i,:),'',[1 1 0]); end
    
    % m.coarse_box = cat(1,m.coarse_box,bbb(1:25,1:4));
    % m.model.x = cat(2,m.model.x, xxx(:,bb(1:25)));

    models2{end+1} = m;
    
    if 0 %%nargin==1
        scaler = [.25 .4 .6 .8];
        for iii = 1:length(scaler)
            
            c1 = round(mean(m.gt_box([2 4])));
            c2 = round(mean(m.gt_box([1 3])));
            
            I2 = m.I;
            I2(c1:m.gt_box(4),m.gt_box(1):m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):c1,m.gt_box(1):m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):m.gt_box(3),c2:m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):m.gt_box(3),m.gt_box(1):c2,:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);

        end
    end

    save(filer,'m');

  end  
end
