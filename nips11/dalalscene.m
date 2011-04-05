function [ws,bs] = dalalscene(X,ids,y,X2,ids2,y2,catnames,targety)
%Train a monolithic classifier for each category

if ~exist('ids','var')
  [X,ids,y,catnames] = sun_initialize('Training_01');
end

if ~exist('ids2','var')
  %Load the testset
  [X2,ids2,y2,catnames] = sun_initialize('Testing_01');
end

myRandomize;
rrr = randperm(length(catnames));
if nargout > 0
  rrr = 1:length(rrr);
  ws = zeros(1984,length(rrr));
  bs = zeros(1,length(rrr));
end
rrr = 111;
for ri = 1:length(rrr)
  targety = rrr(ri);
  
  filer = sprintf('/nfs/baikal/tmalisie/sun/dalals/%05d.mat', ...
                      targety);
  
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

   
  newx = X;
  newy = double(y==targety);
  newy(newy==0) = -1;
  
  SVMC = .01;
  fprintf(1,'training svm\n');
  
  w = mean(X(:,newy==1),2);
  w = w - mean(w(:));
  b = 0;
  
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
  
  res = w'*X - b;
  [aa,bb] = sort(res,'descend');
  stacker1 = create_grid_res(ids(bb),5);
  
  res = w'*X2 - b;
  [aa,bb] = sort(res,'descend');
  stacker2 = create_grid_res(ids2(bb),5);
  stacker1 = cat(2,pad_image(stacker1,20,1),pad_image(stacker2,20,1));
  
  figure(1)
  imagesc(stacker1)
  title(catnames{targety},'interpreter','none')
  drawnow
  
  save(filer,'w','b');
  
  cn = catnames{targety};
  cn = strrep(cn,'/','-');
  finalI = sprintf('/nfs/baikal/tmalisie/sun/dalals/%05d.%s.png', ...
                   targety,cn);
  
  imwrite(stacker1,finalI);
  
  rmdir(filerlock);
  
end
  

