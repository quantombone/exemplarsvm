function [ws,bs] = dalalscene(X,ids,y,X2,ids2,y2,catnames,targety)
%Train a monolithic classifier for each category

mode = 'dalals'
%mode = 'caps'

if ~exist('ids','var')
  [X,ids,y,catnames] = sun_initialize('Training_01');
end

if ~exist('ids2','var')
  %Load the testset
  [X2,ids2,y2,catnames] = sun_initialize('Testing_01');
end

myRandomize;
rrr = randperm(length(catnames));

%rrr = 111;
%rrr = 392;
%rrr = 195;
%rrr = 275;
%rrr = 38;
rrr = 111;
if nargout > 0
  rrr = 1:length(rrr);
  ws = zeros(1984,length(rrr));
  bs = zeros(1,length(rrr));
end

for ri = 1:length(rrr)
  targety = rrr(ri);
  targety
  for index = 1:50
  filer = sprintf('/nfs/baikal/tmalisie/sun/%s/%05d.%d.mat', ...
                      mode,targety,index);  
  if nargout > 0
    fprintf(1,'.');
    res=load(filer);
    ws(:,targety) = res.w;
    bs(targety) = res.b;
    continue;
  end
  
  filerlock = [filer '.lock'];
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end

  fprintf(1,'training svm\n');  
  %[svm_model] = train_dalal(X,y,targety);
  gamma = .01;
  SVMC = .01;
  
  %[svm_model] = train_all(X,y,targety,ids);

  %[svm_model] = train_nonlinear(X,y,targety,svm_model,gamma,SVMC);
  %res2 = mysvmpredict(X2,svm_model);
  %[cap2,p2] = compute_capacity(res2,y2,targety);
  %fprintf(1,'gamma=%.3f, sv = %d, cap =
  %%.3f\n',gamma,svm_model.totalSV,cap2);
  [svm_model] = train_mc(X,y,targety,index,ids);

  res = mysvmpredict(X,svm_model);
  res2 = mysvmpredict(X2,svm_model);
    
  %Compute precision curve and image tile
  [cap1,p1] = compute_capacity(res,y,targety);
  [aa,bb] = sort(res,'descend');
  stacker1 = create_grid_res(ids(bb),5);
  

  [cap2,p2] = compute_capacity(res2,y2,targety);
  [aa,bb] = sort(res2,'descend');
  stacker2 = create_grid_res(ids2(bb),5);

  fprintf(1,'cap1 and cap2 are : %.3f %.3f\n',cap1,cap2);
  
  figure(1)
  clf
  % subplot(1,3,1)
  % imagesc(stacker1)
  % axis image
  % axis off
  % title(['train: ' catnames{targety}],'interpreter','none')

  % subplot(1,3,2)
  % imagesc(stacker2)
  % axis image
  % axis off
  % title(['test: ' catnames{targety}],'interpreter','none')

  
  % subplot(1,3,3)
  plot(p1)
  hold all
  plot(p2)
  legend(sprintf('Train p@k=50: %.3f',cap1),sprintf(['Test p@k=50:' ...
                                 ' %.3f'],cap2))
  xlabel('Top Detection')
  ylabel('Precision')
  title('PR for train/test sets')
  drawnow
  
  save(filer,'svm_model');
  
  cn = catnames{targety};
  cn = strrep(cn,'/','-');
  finalI = sprintf('/nfs/baikal/tmalisie/sun/%s/%05d.%s.%.3f.%.3f.%d.png', ...
                   mode,targety,cn,cap1,cap2,index);
  finalIgraph = sprintf('/nfs/baikal/tmalisie/sun/%s/graph.%05d.%s.%.3f.%.3f.%d.png', ...
                   mode,targety,cn,cap1,cap2,index);
  set(gcf,'PaperPosition',[0 0 4 4]);
  print(gcf,'-dpng',finalIgraph);
  
  stacker1 = cat(2,pad_image(stacker1,20,1),pad_image(stacker2,20,1));
  imwrite(stacker1,finalI);
  figure(2)
  clf
  imagesc(max(0.0,min(1.0,stacker1)))
  drawnow
  
  rmdir(filerlock);
  end
end
  

function [svm_model] = train_dalal(X,y,targety)
%train a monolithic detector

newx = X;
newy = double(y==targety);
newy(newy==0) = -1;

w = mean(X(:,newy==1),2);
w = w - mean(w(:));
b = 0;
SVMC = .01;
wlist = [];
for qqq = 1:4
  [aa,bb] = sort(w'*X,'descend');
  newx = X(:,bb(1:1000));
  newy = double(y(bb(1:1000))==targety);
  newy(newy==0) = -1;
  
  tic
  svm_model = libsvmtrain(newy,newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));
  toc
  fprintf(1,'done\n');
  
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  w = svm_weights';
  b = svm_model.rho;
  if newy(1)==-1
    w = w*-1;
    b = b*-1;
  end
  wlist(:,end+1) = w;
end
res = w'*X-b;

svm_model.w = w;
svm_model.b = b;

function [svm_model] = train_nonlinear(X,y,targety,dalal_model,gamma,SVMC)
%train a monolithic detector

[tmp,bb] = sort(dalal_model.w'*X-dalal_model.b,'descend');
inds = bb(1:3000);

X = X(:,inds);
y = y(inds);
[alpha,beta] = sort(y==targety,'descend');
X = X(:,beta);
y = y(beta);

newx = X;
newy = double(y==targety);
newy(newy==0) = -1;

tic
svm_model = libsvmtrain(newy,newx',sprintf(['-s 0 -t 2 -c' ...
                    ' %f -gamma %f -q'],SVMC,gamma));
toc
fprintf(1,'done\n');



function svm_model = train_mc(X,y,targety,index,ids)
%train a max capacity classifier
goods = (y==targety);
X = X(:,[find(goods); find(~goods)]);
y = y([find(goods); find(~goods)]);
ids = ids([find(goods); find(~goods)]);

newx = X;
newy = double(y==targety);
newy(newy==0) = -1;

w = mean(X(:,newy==1),2);
w = w - mean(w(:));
b = 0;
SVMC = .01;
gamma = 3.0;

wlist = [];
goods = find(y==targety);
bads = find(y~=targety);
NEGBUFFER = 10000;
for qqq = 1:2
  [aa,bb] = sort(w'*X(:,bads)-b,'descend');
  numviols = sum(aa>=-1);
  %NEGBUFFER = min(5000,numviols);

  newx = X(:,[goods; bads(bb(1:NEGBUFFER))]);
 
  newids = ids([goods; bads(bb(1:NEGBUFFER))]);
  newy = double(y([goods; bads(bb(1:NEGBUFFER))])==targety);
  newy(newy==0) = -1;
  [w,b,alphas,pos_inds] = learn_local_capacity(newx,newy,index,SVMC,gamma);
  wlist(:,end+1) = w;
  
  [aa,bb] = sort(w'*X-b,'descend');
  stacker = create_grid_res(ids(bb),5);
  
  stacker_friends = create_grid_res(newids(pos_inds(alphas>0)),5);

  figure(3)
  subplot(1,3,1)
  imagesc(stacker)
  title(sprintf('iteration %d sumalphas=%d',qqq,sum(alphas)))
  
  subplot(1,3,2)
  plot(aa(1:100),'r.')
  title(sprintf('index=%d',index))
  
  subplot(1,3,3)
  imagesc(stacker_friends)
  title(sprintf('iteration %d sumalphas=%d',qqq,sum(alphas)))


  drawnow

end

fprintf(1,'final num alphas is %d\n',sum(alphas));
res = w'*X-b;
svm_model.w = w;
svm_model.b = b;

function [cap,p] = compute_capacity(res,y,targety)
%capacity is average precision across top 100 hits

[aa,bb] = sort(res,'descend');
corr = (y(bb))==targety;
%corr = corr(1:500);
p = (cumsum(corr)./(1:length(corr))');
%rrr = find(cumsum(corr)>=30);
%rrr = rrr(1);
rrr = 100;
%rrr = max(100,rrr);
p = p(1:rrr);
cap = p(50);
return;
%p = (cumsum(corr) - .2*cumsum(corr==0)) ./ (1:length(corr))';
p = (cumsum(corr)./(1:length(corr))').*cumsum(corr) / sum(y==targety);
p = p(1:200);
cap = max(p);

