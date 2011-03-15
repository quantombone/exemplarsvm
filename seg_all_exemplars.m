

function seg_all_exemplars
VOCinit;

addpath(genpath('/nfs/hn22/tmalisie/ddip/segment/'))
addpath(genpath('/nfs/hn22/tmalisie/ddip/cvpr10/'))
addpath(genpath('/nfs/hn22/tmalisie/ddip/util/BSE-1.1/'))
addpath(genpath('/nfs/hn22/tmalisie/ddip/util/graphcuts/'))

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,'motorbike','trainval'),...
                  '%s %d');

ids = ids(gt==1);
myRandomize;
rrr = randperm(length(ids));
ids = ids(rrr);


    % to show stuff
  params.display = 0;
  
  %fix these params to make a unary potential
  params.num_fg = 1;
  params.num_bg = 1;
  params.lambda = -1;
  params.beta = 10;
  params.fg_assignment = -inf;


for i = 1:length(ids)
  curid = ids{i};
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  Ibase = im2double(Ibase);
  
  
    


  % can scan over these vars the quickest since
  % they don't involve recomputation of unary

  for j = 1:length(recs.objects)

    if strcmp(recs.objects(j).class,'motorbike')==0
      continue
    end
    filer = sprintf('/nfs/baikal/tmalisie/iccv11-gcsegs/%s.%05d.mat',...
                      curid,j);
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock)==0)
      continue
    end

    bbox = recs.objects(j).bbox;    
    I = Ibase;
    mask = zeros(size(Ibase,1),size(Ibase,2),1);
    mask(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
    
    % maskbase = mask;
    % scaler = sum(mask(:)) / (100*100)%;/sum(mask(:));
    % scaler = scaler / scaler;
    % scaler = min(scaler,1.0);
    % scaler = 1.0;
    % I = max(0.0,min(1.0,imresize(I,scaler)));
    % mask = imresize(mask,scaler,'nearest');
    
    % [params.pb]= pbGM(I, 2);

    % % params.pb(1:2,:) = 0;
    % % params.pb(end-1:end,:) = 0;
    
    % % params.pb(:,1:2) = 0;
    % % params.pb(:,end-1:end) = 0;
    % % params.pb = (params.pb.*(params.pb>.01));
    

    % % params.Irow = [reshape(I,[],3)];% reshape(RGB2Lab(I),[],3)];
    % % %params.Irow = 
    % % %params.Irow = params.Irow(:,2:3);
  
    % % %the occupied region is always BG and nothing is learned there
    % % params.occupied = logical(zeros(size(params.pb)));

 
    
    
    % % maskstart = mask;
    % % for q = 1:1
    % %   fprintf(1,',');

    % %   idimage = gc3_idimage(mask);
    % %   mask = gc3(I,idimage,params);
    % %   %mask(find(~maskstart)) = 0;
    % % end

    % mask = imresize(mask,[size(Ibase,1),size(Ibase,2)])>.5;
    %mask = params.pb;
    
    seg = MS_seg(I,8,3,size(I,1)*size(I,2)/100);
                                                
    oks = mask+1;
    counts=sparse(seg(:),oks(:),1,max(seg(:)),2);
    counts = counts ./ repmat(sum(counts,2),1,2);
    c1 = counts(:,1); 
    mask = 1-(reshape(c1(seg),size(seg))>.5);

    imagesc(mask)
    drawnow
    save(filer,'mask');
    
    keyboard
    rmdir(filerlock);
    
    % figure(1)
    % clf
    % imagesc(Ibase.*(.1+.9*repmat(mask,[1 1 3])))
    % drawnow
    


    
   
  end
  
  
end
