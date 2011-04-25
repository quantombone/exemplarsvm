function svm_model = train_all(X,y,targety,ids,X2,y2,ids2)
%Train the joint exemplar model (memex model)

saveids = ids;
savey = y;
addpath(genpath('/nfs/hn22/tmalisie/ddip/exemplarsvm/liblinear-1.7/'));

addpath(genpath('/nfs/hn22/tmalisie/ddip/nips2010/'))

if 1
  %if enabled, get wiggles of each image
  goods = find(y==targety);
  for k = 1:length(goods)
    fprintf(1,'.');
    [xs{k},bbs,Is{k}] = get_global_wiggles(convert_to_I(ids{goods(k)}));
    ys{k} = ones(size(xs{k},2),1)*y(goods(k));
  end
  yA = cat(1,ys{:});
  XA = cat(2,xs{:});
  idsA=cat(1,Is{:});

  goods = find(y~=targety);
  X = cat(2,XA,X(:,goods));
  y = cat(1,yA,y(goods));
  ids = cat(1,idsA,ids(goods));

  
end

if 1
  %do pca projection here
  [v,d,mu] = mypca(X,200);
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
  mu = mu*0;
  X = v'*bsxfun(@minus,X,mu);

  if exist('X2','var')
    X2 = v'*bsxfun(@minus,X2,mu);
  end
end

X(end+1,:) = 1;
if exist('X2','var')
  X2(end+1,:) = 1;
end


allgoods = find(y==targety);
saveallbads = find(y~=targety);
%allbads = saveallbads(rand(size(saveallbads))>.5);
N = length(allgoods);

W = X(:,allgoods); 
for i = 1:N
  W(:,i) = W(:,i) - mean(W(:,i));
  %W(:,i) = 0;
  %W(end,i) = 0;
  W(end,i) = -100;
end

A = eye(N);
%A = ones(N);
% d = distSqr_fast(X(:,allgoods),X(:,allgoods));
% for i = 1:N
%   [aa,bb] = sort(d(i,:));
%   A(bb(1:3),i) = 1;
%   %A(i,bb(1:3)) = 1;
% end
% A = A&(A');

y = double(y==targety);
y(y==0) = -1;

meanW = mean(W,2);
allbads = saveallbads(rand(size(saveallbads))>.5);
[aa,bb] = sort(meanW'*X(:,allbads),'descend');
bads = allbads(bb(1:min(length(bb),1000)));

for i = 1:N
  badlen(i) = 500;
end

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
%indlist = [1:N 1:N 1:N];
indlist = [1:N; 1:N; 1:N; 1:N];
%indlist = randperm(N);
%indlist = [indlist; indlist; indlist];
%indlist = indlist(:);
indlist = [1:N];
%indlist = ones(100,1)*1;
%curi = 1;
for iiii = 1:1:length(indlist)
  if mod(iiii,10) == 0
  end
  figure(1)
    
  i = indlist(iiii);
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
  
  oldW = X(:,allgoods(i));
  fprintf(1,'.');
  for repeat = 1:10
    %meanW = W(:,i);
    %[aa,bb] = sort(meanW'*X(:,allbads));
    %bads = allbads(bb(1:1000));
    
    meanW = W(:,i);
    %allbads = saveallbads(rand(size(saveallbads))>.8);
    allbads = saveallbads;
    [aa,bb] = sort(meanW'*X(:,allbads),'descend');
    %numviol = sum(aa>-1);
    if (repeat == 1)
      l = 10000;
    else
      l = 10000;
    end
    %r = randperm(length(allbads));
    %bads = allbads(r(1:min(length(bads),l)));
    bads = allbads(bb(1:min(length(allbads),l)));

    goods = allgoods(find(A((i),:)));
    %notgoods = allgoods(find(~A((i),:)));
    %rrr = randperm(length(goods));
    %goods(rrr(1:min(length(rrr),3))) = [];
    %rrr = randperm(length(notgoods));
    %goods = [goods; notgoods(rrr(1))];
    %goods = unique([goods; allgoods(i)]);
    newg = X(:,[goods]);%; bads]);
    newy = double(y([goods]));%; bads]));
    masker = repmat(reshape(1:64,[8 8]),[1 1 31]);
    superx = [];
    supery = [];
    masker = reshape(1:64,[8 8]); %ones(8,8);
    %masker(1:3,3:6)=1;
    masker = repmat(masker,[1 1 31]);
    %masker = double(masker==1);
    
    if 0
      clear superx;
      clear supery;
      for z = 1:64
        curM = zeros(8*8*31,1);
        curM(masker==z)=1;
        curM(end+1) = 1;
        superx{z} = repmat(curM,1,size(newg,2)).*newg;
        supery{z} = newy;
      end
      superx = cat(2,superx{:});
      supery = cat(1,supery{:});
    end
    newx = X(:,[goods; bads]);%(bb(1:NEGBUFFER))]);
    newids = ids([goods; bads]);%(bb(1:NEGBUFFER))]);
    newy = double(y([goods; bads]));%(bb(1:NEGBUFFER))])==targety);
    newy(newy==0) = -1;
    
    diffx = 1*bsxfun(@minus,X(:,allgoods(i)),X(:,goods));
    diffy = ones(1,size(diffx,2))';
    Krep = length(allgoods)*2;
    Krep = 10;
    superrep = 1;
    newx = cat(2,newx,repmat(diffx,1,Krep),repmat(superx,1,superrep));
    newy = cat(1,newy,repmat(diffy,Krep,1),repmat(supery,superrep, ...
                                                  1));
    
    if 0
    %% pre-aply on something noisy
    j=get_james_bg(iiii);
    noisyI = convert_to_I(j{1});
    localizeparams.thresh = -1000;
    localizeparams.TOPK = 1000; %TOPK_FINAL;
    localizeparams.lpo = 10;
    localizeparams.SAVE_SVS = 1;
  
    models{1}.model.w = reshape(v*W(1:end-1,i),[8 8 31]);
    models{1}.model.b = 0;%-W(end,i);
    models{1}.model.hg_size = [8 8 31];
    models{1}.model.params.sbin = 8;
    [rs,t] = localizemeHOG(noisyI, models, localizeparams);
    try
    xxx = cat(2,rs.support_grid{1}{:});
    xxx = v'*bsxfun(@minus,xxx,mu);
    xxx(end+1,:)=1;
    yyy = -1*ones(size(xxx,2),1);
    newx = cat(2,newx,xxx);
    newy = cat(1,newy,yyy);
    catch
    end
    end
    %wpos = 1*sum(newy==-1) / sum(newy==1);
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
                            sprintf(['-s %s -B -1 -c .1' ...
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
    W(:,i) = model.w';

    %r = W(:,i)'*X;
    %fprintf(1,'from svm: num considered %d\n',sum(r>-1));

    if norm(oldW-W(:,i))<.001
      continue
    end
    
    [aa,bb] = sort(W(:,i)'*X,'descend');  
    yyy = y;
    yyy(yyy<0) = yyy(yyy<0)*10;
    [alpha,beta] = max(cumsum(yyy(bb(1:50))));
    if beta==1
      beta = 2;
    end
    mv = beta;
    
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
    
    if 1
      %dont update A for now
      A(i,:) = 0;
      if length(mv)==0
        
      else
        hits = bb(1:mv);
        [tmp,remaps] = ismember(hits,allgoods);
        remaps = remaps(tmp>0);
        A(i,remaps) = 1;
        A(remaps,i) = 1;
        %[aa,bb] = sort(W(:,i)'*X(:,allgoods),'descend');
        %A(i,bb(1:3)) = 1;
        %A(bb(1:3),i) = 1;
      end
      A(i,i) = 1;
    end
    
    
    
    oldW = W(:,i);   
  end
  
  if 1
    %hinge = @(x)max(1-x,0);
    %find self-losses function
    %r=W'*X(:,allgoods);
    %h=r;
    figure(1)
    subplot(3,2,1)
    imagesc(hogpic)
    title('hogpic')
    axis off
    axis image
    
    subplot(3,2,2)
    imagesc(nhogpic)
    title('negative hog pic')
    axis off
    axis image


    

  end
    
  
  if 1

    r = W(:,i)'*X;
    %r = W'*X;
    %sigmoid = @(x)(1./(1+exp(-x)));
    %r = sum(sigmoid(r),1);
    [aa,bb] = sort(r,'descend');
    superind = find(bb==allgoods(i));
    relatedinds = find(y(bb)==1);
    
    subplot(3,2,3)    
    stacker1 = create_grid_res(ids(bb),5,[100 100],superind, ...
                               relatedinds);
    %stacker1 = zeros(100,100,3);
    imagesc(stacker1)
    axis off
    axis image
    title(sprintf('trainingset: %.3f %.3f %.3f %.3f',aa(1),aa(2),aa(3),aa(4)))
    
    %end
    
    subplot(3,2,5)
    stacker = show_hog_stacker(bsxfun(@plus,v*X(1:end-1,bb),mu),curw);
    %stacker = show_hog_stacker(X(1:end-1,bb),curw);
    imagesc(stacker)
    axis image
    axis off
    title('train Scoremaps');

    
    %r = W(:,i)'*X2;
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
    r = W(:,i)'*curX;
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
    subplot(3,2,4)
    %plot(y(bb),'r.');
    imagesc(stacker2)
    axis off
    axis image
    title(sprintf('%s: %.3f %.3f %.3f %.3f',curtitle,aa(1),aa(2),aa(3),aa(4)))
        
    subplot(3,2,6)
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
  
  
  set(gcf,'PaperPosition',[0 0 20 20]);
  print(gcf,'-dpng',sprintf(['/nfs/baikal/tmalisie/sun/dalals/memex/chunky.%d.' ...
                                                                    '%05d.png'],targety,iiii));
  
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
  stacker{qqq} = mys; %padarray(mys,[1 1 0]);
end
    
stacker = create_grid_res(stacker,5,[8 8]);
stacker = stacker(:,:,1);
