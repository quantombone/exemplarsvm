function [m] = do_svm(m,mining_params)
%Perform SVM learning for a single exemplar model, we assume that
%the exemplar has a set of detections loaded in m.model.svxs and m.model.svbbs
%Durning Learning, we can apply some pre-processing such as PCA or
%dominant gradient projection

%Tomasz Malisiewicz (tomasz@cmu.edu)

if ~isfield(m.model,'mask') | length(m.model.mask)==0
  m.model.mask = logical(ones(numel(m.model.w),1));
end

if length(m.model.mask(:)) ~= numel(m.model.w)
  m.model.mask = repmat(m.model.mask,[1 1 features]);
  m.model.mask = logical(m.model.mask(:));
end

%% look into the object inds to figure out which subset of the data
%% is actually hard negatives for mining
if mining_params.extract_negatives == 1
  [negatives,vals,pos,m] = find_set_membership(m);
  
  xs = m.model.svxs(:, [negatives]);
  bbs = m.model.svbbs([negatives],:);
    
else
  xs = m.model.svxs;
  bbs = m.model.svbbs;
end

MAXSIZE = 2000;
if size(xs,2) >= MAXSIZE
  HALFSIZE = MAXSIZE/2;
  %NOTE: random is better than top 5000
  r = m.model.w(:)'*xs;
  [tmp,r] = sort(r,'descend');
  r1 = r(1:HALFSIZE);
  
  r = HALFSIZE+randperm(length(r((HALFSIZE+1):end)));
  r = r(1:HALFSIZE);
  r = [r1 r];
  xs = xs(:,r);
  bbs = bbs(r,:);
end


if 1
  %old method
  
  superx = cat(2,m.model.x,xs);
  supery = cat(1,ones(size(m.model.x,2),1),-1*ones(size(xs,2),1));
  
  
  spos = sum(supery==1);
  sneg = sum(supery==-1);

  wpos = 50;
  wneg = 1;

  if mining_params.BALANCE_POSITIVES == 1
    fprintf(1,'balancing positives\n');
    wpos = 1/spos;
    wneg = 1/sneg;
    wpos = wpos / wneg;
    wneg = wneg / wneg;
  end
  
  A = eye(size(superx,1));
  mu = zeros(size(superx,1),1);
  
  if mining_params.DOMINANT_GRADIENT_PROJECTION == 1  
    A = get_dominant_basis(reshape(mean(m.model.x(:,1),2), ...
                                   m.model.hg_size),...
                           mining_params.DOMINANT_GRADIENT_PROJECTION_K);
    
    
    A2 = get_dominant_basis(reshape(mean(superx(:,supery==-1),2), ...
                                    m.model.hg_size),...
                            mining_params ...
                            .DOMINANT_GRADIENT_PROJECTION_K);
    A = [A A2];
  elseif mining_params.DO_PCA == 1
    [A,d,mu] = mypca(superx,mining_params.PCA_K);
  elseif mining_params.A_FROM_POSITIVES == 1
    A = [superx(:,supery==1)];
    cursize = size(A,2);
    for qqq = 1:cursize
      A(:,qqq) = A(:,qqq) - mean(A(:,qqq));
      A(:,qqq) = A(:,qqq)./ norm(A(:,qqq));
    end
        
    %% add some ones
    A(:,end+1) = 1;
    A(:,end) = A(:,end) / norm(A(:,end));
  end
  
  newx = bsxfun(@minus,superx,mu);
  newx = newx(logical(m.model.mask),:);
  newx = A(m.model.mask,:)'*newx;
  
  fprintf(1,' -----\nStarting SVM dim=%d... s+=%d, s-=%d ',...
          size(newx,1),spos,sneg);
  starttime = tic;
  
  svm_model = libsvmtrain(supery, newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'], mining_params.SVMC, wpos));
  
  if length(svm_model.sv_coef) == 0
    %learning had no negatives
    wex = m.model.w;
    b = m.model.b;
    fprintf(1,'reverting to old model...\n');
  else
    
    %convert support vectors to decision boundary
    svm_weights = full(sum(svm_model.SVs .* ...
                           repmat(svm_model.sv_coef,1, ...
                                  size(svm_model.SVs,2)),1));
    
    wex = svm_weights';
    b = svm_model.rho;
    
    if supery(1) == -1
      wex = wex*-1;
      b = b*-1;    
    end
    
    %% project back to original space
    b = b + wex'*A(m.model.mask,:)'*mu(m.model.mask);
    wex = A(m.model.mask,:)*wex;
    
    wex2 = zeros(size(superx,1),1);
    wex2(m.model.mask) = wex;
    
    wex = wex2;
    
    %% issue a warning if the norm is very small
    if norm(wex) < .00001
      fprintf(1,'learning broke down!\n');
    end  
  end
  
  maxpos = max(wex'*m.model.x - b);
  fprintf(1,' --- Max positive is %.3f\n',maxpos);
  
  fprintf(1,'took %.3f sec\n',toc(starttime));
end %end old method


m.model.w = reshape(wex, size(m.model.w));
m.model.b = b;

r = m.model.w(:)'*m.model.svxs - m.model.b;
svs = find(r >= -1.0000);

%KEEP 3#SV vectors (but at most max_negatives of them)
total_length = ceil(mining_params.beyond_nsv_multiplier*length(svs));
total_length = min(total_length,mining_params.max_negatives);

[alpha,beta] = sort(r,'descend');
svs = beta(1:min(length(beta),total_length));
m.model.svxs = m.model.svxs(:,svs);
m.model.svbbs = m.model.svbbs(svs,:);
%TODO: Note, we are keeping vectors, but some are actually for
%validation, some for training...


