function svm_model = train_all(saveX,savey,targety,saveids,saveX2,y2,ids2)
%Train the joint exemplar model (memex model)

% norms = sqrt(sum(saveX.^2,1));
% for i = 1:size(saveX,2)
%   saveX(:,i) = saveX(:,i) / norms(i);
% end

% norms = sqrt(sum(saveX2.^2,1));
% for i = 1:size(saveX2,2)
%   saveX2(:,i) = saveX2(:,i) / norms(i);
% end


X = saveX;
y = savey;
ids = saveids;

addpath(genpath('/nfs/hn22/tmalisie/ddip/exemplarsvm/liblinear-1.7/'));
addpath(genpath('/nfs/hn22/tmalisie/ddip/nips2010/'))

if 0
filer = sprintf('/nfs/baikal/tmalisie/sunpcacrops/%05d.mat',targety);
if ~fileexists(filer)
  [v,d,mu] = mypca(X,10);
  %if enabled, get wiggles of each image
  goods = find(y==targety);
  for k = 1:length(goods)
    fprintf(1,'.');
    [xs{k},bbs{k},Is{k}] = ...
        get_global_wiggles(convert_to_I(ids{goods(k)}));
    
    newx = v'*bsxfun(@minus,xs{k},mu);
    newxx = bsxfun(@plus,v*newx,mu);
    [alpha,beta] = sort((sum((xs{k} - newxx).^2,1)));
    % I = convert_to_I(ids{goods(k)});
    % scaler = 300/max(size(I,1),size(I,2));
    % I = imresize(I,scaler);
    % I = max(0.0,min(1.0,I));

    % figure(2)
    % clf
    % for q = 1:9
    %   subplot(3,3,q)
    %   imagesc(I)
    %   plot_bbox(bbs(beta(q),:))
    %   axis image
    %   axis off
    %   %imagesc(Is{k}{beta(q)})
    %   title(num2str(alpha(q)))
    % end
    % pause
    ys{k} = ones(size(xs{k},2),1)*y(goods(k));
    
    finders = beta(1:10);
    xs{k} = xs{k}(:,finders);
    bbs{k} = bbs{k}(finders,:);
    ys{k} = ys{k}(finders);
    Is{k} = Is{k}(finders);
  end
  
  save(filer,'xs','bbs','ys');
else
  load(filer);
  goods = find(y==targety);
  Is = cell(1,length(bbs));
  for i= 1:length(goods)
    I = convert_to_I(ids{goods(i)});
    fprintf(1,',');
    scaler = 300/max(size(I,1),size(I,2));
    I = imresize(I,scaler);
    I = max(0.0,min(1.0,I));

    for z = 1:size(bbs{i},1)
      bb = bbs{i}(z,:);
      curI = I(bb(2):bb(4),bb(1):bb(3),:);
      curI = imresize(curI,[200 200]);
      curI = max(0.0,min(1.0,curI));
      Is{i}{z}=curI;
    end
  end

end

  yA = cat(1,ys{:});
  XA = cat(2,xs{:});
  idsA=cat(1,Is{:});
  idsA = reshape(idsA,[],1);
 

  goods = find(y~=targety);
  saveX = cat(2,XA,X(:,goods));
  savey = cat(1,yA,y(goods));
  saveids = cat(1,idsA,ids(goods));
  
  X = saveX;
  y = savey;
  ids = saveids;

end
 
if 0
  %do pca projection here
  [v,d,mu] = mypca(X,100);
  %mu = mean(X,2);
  %v = eye(length(mu),length(mu));
  %saveX = X;
  X = v'*bsxfun(@minus,X,mu);

  if exist('X2','var')
    X2 = v'*bsxfun(@minus,X2,mu);
  end
else
  v = eye(size(X,1));
  mu = mean(X,2);
  %better with mu = 0
  mu = mu*0;
  X = v'*bsxfun(@minus,X,mu);

  if exist('X2','var')
    X2 = v'*bsxfun(@minus,X2,mu);
  end
end

allgoods = find(y==targety);
saveallbads = find(y~=targety);

N = length(allgoods);

% W = X(:,allgoods); 
% for i = 1:N
%   W(:,i) = W(:,i) - mean(W(:,i));
% end


%A = eye(N);
A = ones(N);

% d = distSqr_fast(X(:,allgoods),X(:,allgoods));
% for i = 1:N
%   [aa,bb] = sort(d(i,:));
%   A(bb(1:3),i) = 1;
%   %A(i,bb(1:3)) = 1;
% end
% A = A&(A');

y = double(y==targety);
y(y==0) = -1;

%allbads = saveallbads(rand(size(saveallbads))<.9);
%bads = allbads;
%[aa,bb] = sort(meanW'*X(:,allbads),'descend');
%bads = allbads(bb(1:min(length(bb),1000)));

% Is = cell(N,1);
% for i = 1:N
%   [a,b] = fileparts(ids{allgoods(i)});
%   iconfile = sprintf('/nfs/baikal/tmalisie/sun/icons/%s.png',b);
%   if fileexists(iconfile);
%     I = imread(iconfile);
%   else
%     I = convert_to_I(ids{allgoods(i)});
%     I = max(0.0,min(1.0,imresize(I,[200 200])));
%     imwrite(I,iconfile);
%   end
%   Is{i} = I;
% end

indlist = [randperm(N) randperm(N) randperm(N) randperm(N) ...
           randperm(N)];
indlist = [1:N 1:N 1:N];
%indlist = [1:N; 1:N; 1:N; 1:N];
%indlist = randperm(N);
%indlist = [indlist; indlist; indlist];
%indlist = indlist(:);
%indlist = [1:N];
%indlist = ones(100,1)*1;
%curi = 1;

%W(end+1,:) = 0;
for iiii = 1:1:length(indlist)
  if mod(iiii,10) == 0
  end
  figure(1)
  clf
  ha = tight_subplot(3, 2, .05, ...
                     .05, .01);
  i = indlist(iiii);
  %fprintf(1,'hack ones set to A\n');
  %A = ones(size(A));
  %% do pca-like projection from DOG template
  %v = get_dominant_basis(reshape(saveX(:,allgoods(i)),[8 8 31]),1);
  v = eye(8*8*31);
  %v = saveX(:,allgoods);
  %for qqq = 1:size(v,2)
  %  v(:,qqq) = v(:,qqq) - mean(v(:,qqq));
  %end
  %v = v.*(v>0);
  %v2 = get_dominant_basis(reshape(mean(saveX,2),[8 8 31]),4);
  %v = [v v2];
  mu = zeros(size(saveX,1),1);
  
  X = v'*bsxfun(@minus,saveX,mu);
  
  if exist('saveX2','var')
    X2 = v'*bsxfun(@minus,saveX2,mu);
  end
  
  X(end+1,:) = 1;
  if exist('X2','var')
    X2(end+1,:) = 1;
  end

  
  %i = indlist(1);  
  %i = curi;

  % mining_params = get_default_mining_params;
  % models{1}.model.x = reshape(X(1:(end-1),i),[8 8 31]);

  % models{1}.model.w = models{1}.model.x - mean(models{1}.model.x(: ...
  %                                                 ));
  % models{1}.model.x = models{1}.model.x(:);
  % models{1}.models_name = 'dalalscene';
  % models{1}.model.wtrace = cell(0,1);
  % models{1}.model.btrace = cell(0,1);
  % models{1}.model.b = -W(end,1);
  % models{1}.model.nsv = [];
  % models{1}.model.svids = [];
  % models{1}.model.hg_size = size(models{1}.model.w);
  % models{1}.model.params.sbin = 20;
  % bg = get_pascal_bg('trainval');
  % bg = bg(1:100);
  % mining_queue = initialize_mining_queue(bg);
  % for q = 1:2
  % [models, mining_queue] = ...
  %     mine_negatives(models, mining_queue, bg, mining_params, q);
  % end
  
  %[xs,bs,Is] = get_global_wiggles(convert_to_I(ids{allgoods(i)}));
  %xs(end+1,:) = 1;
  
  bg = get_james_bg(1000);
  mining_queue = initialize_mining_queue(bg);
  
  oldW = X(:,allgoods(i));
  fprintf(1,'.');
  
  curW = [];
  for repeat = 1:3
    
    %if first iteration, take random negatives
    if length(curW) == 0
      allbads = saveallbads(rand(size(saveallbads))<.8);
      bads = allbads;
      curW = X(:,allgoods(i));
      curW = curW - mean(curW(:));
    else
      allbads = saveallbads(rand(size(saveallbads))<.9);
      [aa,bb] = sort(curW'*X(:,allbads),'descend');
      LEN = 5000;
      bads = allbads(bb(1:min(length(allbads),LEN)));
    end
    
    goods = allgoods(find(A((i),:)));

    newg = X(:,[goods]);%bads]);
    newy = double(y([goods]));%; bads]));
    masker = repmat(reshape(1:64,[8 8]),[1 1 31]);
    superx = [];
    supery = [];
    masker = reshape(1:64,[8 8]); %ones(8,8);

    masker = repmat(masker,[1 1 31]);
    %masker = double(masker==1);
    
    if 0
      %curX = X(:,allgoods(i));
      %curX = ((reshape(W(1:end-1,i).*curX(1:end-1),[8 8 31])));
      %curS = (sum(curX,3));
      %signmap = 1*double(sign(curS));
      %signmap = signmap*0+1;
      
      if 0 %exist('oldsignmap','var')
        potmap1 = zeros(64,1);
        potmap2 = zeros(64,1);
        for z = 1:64
          curM = zeros(8*8*31,1);
          curM(masker==z)=1;
          curM(end+1) = 1;
          sx= W(:,i)'*(repmat(curM,1,size(newg,2)).*newg);
          potmap1(z) = sum(hinge(sx));
          potmap2(z) = sum(hinge(-sx));
        end
      end
        
      %signmap = -1*ones(8,8);
      %signmap(3:7,3:7) = 1;
      %signmap = signmap*.1;
      
      clear superx;
      clear supery;
      for z = 1:64
        curM = zeros(8*8*31,1);
        curM(masker==z)=1;
        curM(end+1) = 1;
        superx{z} = 100*(repmat(curM,1,size(newg,2)).*bsxfun(@minus,X(:,allgoods(i)),newg));
        supery{z} = newy*0+1;
      end
      superx = cat(2,superx{:});
      supery = cat(1,supery{:});
      
      %oldsignmap = signmap;
    end
    
    newx = X(:,[goods; bads]);
    newids = ids([goods; bads]);
    newy = double(y([goods; bads]));
    newy(newy==0) = -1;
    
    %fprintf(1,'turned off selfmax\n');
    diffx = 1*bsxfun(@minus,X(:,allgoods(i)),X(:,goods));
    diffy = ones(1,size(diffx,2))';
    Krep = length(allgoods)*2;
    superrep = 1;
    
    %fprintf(1,'warning dropping global w''x\n')
    %newx = cat(2,repmat(diffx,1,Krep),repmat(superx,1,superrep));
    %newy = cat(1,repmat(diffy,Krep,1),repmat(supery,superrep, ...
    %                                              1));

    newx = cat(2,newx,repmat(diffx,1,Krep),repmat(superx,1,superrep));
    newy = cat(1,newy,repmat(diffy,Krep,1),repmat(supery,superrep, ...
                                                  1));
    
    if 0
    %% mine stuff from james

    mining_params = get_default_mining_params;
    mining_params.detection_threshold = -1.05;
    mining_params.MAX_IMAGES_BEFORE_SVM = 3;
    %noisyI = convert_to_I(j{1});
    
    models{1}.model.w = reshape(v*curW(1:end-1),[8 8 31]);
    models{1}.model.b = -(curW(end) - curW(1:end-1)'*v'*mu);
    models{1}.model.hg_size = [8 8 31];
    models{1}.model.params.sbin = 8;
    
    [hn, models, mining_queue, mining_stats] = ...
        load_hn_fg(models, mining_queue, bg, mining_params);

    xxx = cat(2,hn.xs{1});
    xxx = v'*bsxfun(@minus,xxx,mu);
    xxx(end+1,:)=1;
    
    rrr = randperm(size(xxx,2));
    rrr = rrr(1:round(length(rrr)*.2));
    heldoutxxx = xxx(:,rrr);
    xxx(:,rrr) = [];
    yyy = -1*ones(size(xxx,2),1);
    newx = cat(2,newx,xxx);
    newy = cat(1,newy,yyy);
   
    % localizeparams.thresh = -1000;
    % localizeparams.TOPK = 1000; %TOPK_FINAL;
    % localizeparams.lpo = 20;
    % localizeparams.SAVE_SVS = 1;
    %[s,t] = localizemeHOG(noisyI, models, localizeparams);
    % if length(rs.support_grid{1})>0
    %   xxx = cat(2,rs.support_grid{1}{:});
    %   xxx = v'*bsxfun(@minus,xxx,mu);
    %   xxx(end+1,:)=1;
    %   yyy = -1*ones(size(xxx,2),1);
    %   newx = cat(2,newx,xxx);
    %   newy = cat(1,newy,yyy);
    % end
    end
    

    wpos = 1.0;
    
    %if repeat == 1
    %  wpos = 0;
    %end
    %wpos = 0, means one-clas SVM!
    
    %from liblinear readme
    % 0 -- L2-regularized logistic regression (primal)
    % 1 -- L2-regularized L2-loss support vector classification (dual)
    % 2 -- L2-regularized L2-loss support vector classification (primal)
    % 3 -- L2-regularized L1-loss support vector classification (dual)
    % 4 -- multi-class support vector classification by Crammer and Singer
    % 5 -- L1-regularized L2-loss support vector classification
    % 6 -- L1-regularized logistic regression
    % 7 -- L2-regularized logistic regression (dual)
    
    % masker = ones(8,8,1);
    % booler = logical(repmat(masker,[1 1 31]));
    % booler = booler(:);
    % booler1 = booler;
    % booler1(end+1) = 1;

    %modes = {'0','1','2','3','5','6','7'}
    modes = '2';
    starter = tic;
    fprintf(1,'starting liblinear:');
    
    %masker = masker(:);
    %masker(end+1) = 1;
    %masker = ones(size(newx,1),1);
    
    model = liblinear_train(newy, sparse(newx)', ...
                            sprintf(['-s %s -B -1 -c 1' ...
                    ' -w1 %.3f -q'],modes,wpos));
    fprintf(1,'Done in %.3f\n',toc(starter));
    curw = model.w(1:end-1)';
    %realw = zeros(length(masker),1);
    %realw(find(masker)) = model.w;
    %model.w = realw';
    %curw = model.w(1:end-1)';

    savew = curw;
    curw = v*curw;
    hogpic=(HOGpicture(reshape(curw,[8 8 31])));
    nhogpic=(HOGpicture(reshape(-curw,[8 8 31])));
        
    
    %figure(333+q)
    %imagesc(hogpic)
    %title(modes{q})
    %drawnow
    %end
    
    %resultw = zeros(8,8,31);
    %resultw(booler) = model.w(1:end-1);
    %resultw = [resultw(:); model.w(end)];
    %W(:,i) = resultw(:);
    %apply model to all of data
    curW = model.w';

    %r = W(:,i)'*X;
    %fprintf(1,'from svm: num considered %d\n',sum(r>-1));

    if norm(oldW-curW)<.001
      continue
    end
    
    if ~exist('heldoutxxx','var')
      heldoutxxx = [];
    end
    
    %find threshold tweaked to maximize gain when negatives count
    %as -1 points and positives count as +1 points
    newX = cat(2, X, heldoutxxx);
    newY = cat(1, y, -1*ones(size(heldoutxxx,2),1));
    [aa,bb] = sort(curW'*newX,'descend');  
    yyy = newY;
    yyy(yyy<0) = -1;
    [alpha,beta] = max(cumsum(yyy(bb(1:1000))));
    if beta==1
      beta = 2;
    end
    mv = beta;

    fprintf(1,'mv is %d\n',mv);
    
    if 0
    %rescale such that sigmoid(w'*x)=.95 on self a,d 
    %sigmoid(w'*x)=.5 on max capacity threshold
    wmat = [aa(1) 1; aa(beta) 1];
    sigmoidinv = @(x)(-log(1./x-1));
    bmat = sigmoidinv([.95 .5]');
    if det(wmat) == 0
      W(:,i) = 0;
    else
      betas = inv(wmat)*bmat;
      W(1:end-1, i) = betas(1)*W(1:end-1,i);
      W(end, i) = betas(1)*W(end,i) + betas(2);
    end
    end
    
    if 1
      %update A for now
      A(i,:) = 0;
      if length(mv)==0
      else
        hits = bb(1:mv);
        [tmp,remaps] = ismember(hits,allgoods);
        remaps = remaps(tmp>0);
        A(i,remaps) = 1;
        A(remaps,i) = 1;
      end
      A(i,i) = 1;
    end
    
   
    oldW = curW;
  end
  
  if 1
    %hinge = @(x)max(1-x,0);
    %find self-losses function
    %r=W'*X(:,allgoods);
    %h=r;
    figure(1)
    axes(ha(1));
    imagesc(hogpic)
    title('HOG positive part')
    axis off
    axis image
    
    axes(ha(2));
    imagesc(nhogpic)
    title('HOG negative part')
    axis off
    axis image


    

  end
    
  
  if 1

    r = curW'*X;
    rtrain = r;
    %r = W'*X;
    %sigmoid = @(x)(1./(1+exp(-x)));
    %r = sum(sigmoid(r),1);
    [aa,bb] = sort(r,'descend');
    superind = find(bb==allgoods(i));
    relatedinds = find(y(bb)==1);
    

    axes(ha(3));
    stacker1 = create_grid_res(ids(bb),5,[100 100],superind, ...
                               relatedinds);
    %stacker1 = zeros(100,100,3);
    imagesc(stacker1)
    axis off
    axis image
    title(sprintf('trainingset: %.3f %.3f %.3f %.3f',aa(1),aa(2),aa(3),aa(4)))
    
    %end
    

    axes(ha(5));    
    stacker = show_hog_stacker(bsxfun(@plus,v*X(1:end-1,bb),mu),curw);
    %stacker = show_hog_stacker(X(1:end-1,bb),curw);
    imagesc(stacker)
    axis image
    axis off
    title('train Scoremaps');

    
    if exist('X2','var')
      curX = X2;
      cury = y2;
      curids = ids2;
      curtitle = 'testset';      
    else
      curX = X;
      cury = y;
      curids = ids;
      curtitle = 'trainset';
    end
    r = curW'*curX;
    %r = sigmoid(r);
    
    %resser = zeros(1,size(r,2));
    %for jjj = 1:size(r,2)
    %  resser(jjj) = r(:,jjj)'*A*r(:,jjj);
    %end
    %r = resser;
    %r = max(r,[],1);
    
    [aa,bb] = sort(r,'descend');
    superind = -1;
    relatedinds = find(cury(bb)==targety);
    
    stacker2 = create_grid_res(curids(bb),5,[100 100],superind, ...
                               relatedinds);
    %stacker2 = zeros(300,300,3);

    axes(ha(4));    
    %plot(y(bb),'r.');
    imagesc(stacker2)
    axis off
    axis image
    title(sprintf('%s: %.3f %.3f %.3f %.3f',curtitle,aa(1),aa(2),aa(3),aa(4)))
        
    axes(ha(6));    

    stacker = show_hog_stacker(bsxfun(@plus,v*curX(1:end-1,bb),mu),curw);
    imagesc(stacker)
    axis image
    axis off
    title('Test Scoremaps');    
    drawnow

    %[aa,bb] = sort(W(:,i)'*X(:,allgoods),'descend');
    %rrr = randperm(2);
    %curi = bb(rrr(1)+1);
  end
  
  %r=W(:,i)'*xs;
  %figure(44)
  %plot(r)
  %drawnow

  %[A,h] = get_A(W,X,y,allgoods);
  
  
  set(gcf,'PaperPosition',[0 0 8 15]);
  print(gcf,'-dpng',sprintf(['/nfs/baikal/tmalisie/sun/dalals/memex/chunky.%d.' ...
                                                                    '%05d.png'],targety,iiii));

  figure(3443)
  clf
  [r1,order] = sort(rtrain,'descend');
  r1 = r1(1:250);
  plot(r1,'r.')
  hold all
  goods = find(y(order(1:250))==targety)
  plot(goods,r1(goods),'ko');
  [r2,order] = sort(r,'descend');
  r2 = r2(1:250);
  plot(r2,'b.')
  goods = find(cury(order(1:250))==targety)
  plot(goods,r2(goods),'mo');
  %legend('train','trainpos','test','testpos')
  drawnow
  
  
  %imwrite(I,sprintf('/nfs/baikal/tmalisie/sun/dalals/memex/g.%d.%05d.png',targety,iiii));
  if 0
  relatedinds = allgoods(find(A(i,:)));
    for j = 1:length(relatedinds)
    cur = relatedinds(j);
    [xs,bs,Is] = get_global_wiggles(convert_to_I(saveids{cur}));
    xs(end+1,:) = 1;
    
    r = W(:,i)'*xs;
    [alpha,beta] = max(r);
    X(:,cur) = xs(:,beta);
    ids{cur} = Is{beta};
    fprintf(1,'got beta of %d\n',beta);
  end
end



end



svm_model.W = W;



return;



W = X;
N = size(X,2);
%A = randn(N,N);
%A = eye(N,N) + randn(N,N);

%A = .1*imresize(eye(4),[N N],'nearest');
A = eye(N);
% for K = 1:3
% A = eye(N,N);
if 0
ddd = distSqr_fast(X,X);
K = 3;
 for i = 1:N
   [a,b] = sort(ddd(i,:));
   A(i,b(1:K))=1;
   A(b(1:K),i)=1;
 end
 end
% A = A + eye(N,N);
% 
% end

bestR = omega(A,X);
bestA = A;

options = optimset('MaxIter',20,'Display','Iter');

for i = 1:100000 %size(X,2)
  differ = .1*sign(randn(size(A)));
  differ(rand(size(differ))>.01)=0;
  differ = (differ+differ')/2;

  A2 = bestA + differ;
  r2 = omega(A2,X);
  if (r2 < bestR)
    bestR = r2;
    bestA = A2;
    figure(1)
    imagesc(A2)
    
    W = X*bestA;
    inds = [1 5 10];
    c = 1;
    for q = 1:length(inds)
      w = W(:,inds(q));
      I = convert_to_I(ids{inds(q)});
      subplot(length(inds),2,c);
      imagesc(I)
      subplot(length(inds),2,c+1);
      imagesc(HOGpicture(reshape(w,[8 8 31])))
      c = c + 2;
    end
    drawnow
    fprintf(1,'bestR is %.3f\n',bestR);
  end
end

function [res,grad] = omega(A,X)
W = X*A;
N = size(W,2);
resmat = W'*X;
[aa,bb] = sort(resmat,2,'descend');
res = 0;
lambda = [100 0];
for i = 1:N
  for j = 1:N
      rij = bb(i,j);
      rji = bb(j,i);
      res = res + L(rij)*L(rji)*(resmat(i,j)-resmat(j,i)).^2;
      res = res + lambda(1)*hinge(resmat(i,i)-resmat(i,j));
  end
end
%res = res + lambda(2)*norm(resmat);

function w=L(r)
w = double(r<=10);
%w = 1./r;

function y=hinge(x)
y = max(1-x,0.0);

function [A,h] = get_A(W,X,y,allgoods)
hinge = @(x)max(1-x,0);
%find self-losses function
r=W'*X(:,allgoods);
h=(hinge(r));
gamma = 1;
A = double(h+h'<2*gamma);

%diag always turned on
A(find(eye(size(A))))=1;

function y = sigmoid(x)
y = 1./(1+exp(-1*x));

function stacker = show_hog_stacker(X,curw)
myw = reshape(curw,[8 8 31]);
stacker = cell(1,25);
for qqq = 1:25
  myX = reshape(X(:,(qqq)),[8 8 31]);
  mys = sum(myX.*myw,3);
  stacker{qqq} = padarray(mys,[1 1 0]);
end
    
stacker = create_grid_res(stacker,5,[10 10]);
stacker = stacker(:,:,1);
