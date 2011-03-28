function [w,b,r,pos_inds] = learn_local_rank_capacity(x,y,index,SVMC,gamma,g,m)
%maximum-capacity learning (alpha,beta) updates 
%ranker-mode

pos_inds = find(y==y(index));
neg_inds = find(y~=y(index));
if ~exist('g','var')
  % fprintf(1,'Creating G from euclidean distance\n');
  % ds = distSqr_fast(x(:,index),x(:,y==y(index)));
  % ds = ds / mean(ds(:));
  % g = exp(-1.0*ds)';
  g = -1*ones(size(x,2),1);
  g(1:length(pos_inds))=1;
end

%initialize r to be itself only
r = [index];


oldgoods = [];
for k = 1:3
  fprintf(1,'#');
  
  newy = y([r; neg_inds]);
  newx = x(:,[r; neg_inds]);
  newg = g([r; neg_inds]);
  
  %% ranking by margin of .1
  diff1 = [];%diff(10*x(:,r),1,2);
  %diff2 = bsxfun(@minus,x(:,r(end)),x(:,neg_inds));
  
  %the top 1 should defeat others by margin of 1
  diff3 = bsxfun(@minus,x(:,r(1)),x(:,pos_inds));
  
  %diff4 = [];
  diff4 = bsxfun(@minus,x(:,r(1)),x(:,r(2:end)));
  
  newd = cat(2,diff1,diff3,diff4);
  newy = ones(1,size(newd,2))';
  %newd = newd(:,[]);
  %newy = newy([]);

  mx = [];
  if exist('m','var')
    mx = m.model.x;
  end
  
  newd = cat(2,newd,mx,x(:,r),x(:,neg_inds));
  newy = cat(1,newy,ones(size(mx,2),1),ones(length(r),1),-ones(length(neg_inds),1));


  
  if 0
    
    %meanx = mean(x,2);
    %newd = bsxfun(@minus,meanx,newd);
    svm_model = liblinear_train(newy, sparse(newd)', ...
                                sprintf(['-s 3 -B -1 -c' ...
                    ' %f -q'],SVMC));
    
    w = svm_model.w';%(1:end-1)';
    %w = w - meanx;
    b = 0;
    %b = -svm_model.w(end)*100;



  end
  
if 1

  %% had -s 2
  svm_model = svmtrain(newy,newd',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));
  
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  w = svm_weights';
  b = svm_model.rho;
  

end

  if y(1)==-1
    w = w*-1; 
    b = b*-1;
  end

  if 1
    %optimize ranking vector r
    %hinge = @(x)max(1-x,0.0);
    
    %[alpha,beta] = sort(w'*newx(:,pos_inds)-b,'descend');
    %[aa,bb] = sort(beta);
    %rankscores = 1./bb;
    loss_term = ((w'*x(:,pos_inds)-b));%; + 20*newg'.*(newg>.5)';% +
                                                         
    %loss_term(neg_inds) = 0;
    loss_term(pos_inds==index) = -100;
    [aa,bb] = sort(loss_term,'descend');
    r = [index pos_inds(bb(1:9))']';%(length(r)+1))';
    
    lt = w'*x-b;
    
    
    
    
  end
    
  
end  





