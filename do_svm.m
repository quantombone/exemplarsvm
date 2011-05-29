function [m] = do_svm(m,mining_params)
%Perform SVM learning for a single exemplar model, we assume that
%the exemplar has a set of detections loaded in m.nsv and m.svids 
%Durning Learning, we can apply some pre-processing such as PCA or
%dominant gradient projection

%Tomasz Malisiewicz (tomasz@cmu.edu)

if 0
  fprintf(1,'using liblinear, might break\n');
  %playing with liblinear here
  %SVMC = .01;
  %mining_params.SVMC = 1;
  addpath(genpath('/nfs/hn22/tmalisie/ddip/exemplarsvm/liblinear-1.7/'));
  model = liblinear_train(supery, sparse(superx)', sprintf(['-s 3 -B 1 -c' ...
                    ' %f'],mining_params.SVMC));
  wex = model.w(1:end-1)';
  b = -model.w(end);
  

  % r = wex(:)'*superx;
  % r(supery==-1) = 10000;
  % [aa,bb] = sort(r,'ascend');
  % supery2 = supery;
  % supery2(bb(1:100)) = [];
  % superx2 = superx;
  % superx2(:,bb(1:100)) = [];
  
  % lmodel = liblinear_train(supery2, sparse(superx2)', sprintf(['-s 3 -B 1 -c' ...
  % ' %f'],mining_params.SVMC));
  % wex = model.w(1:end-1)';
  % b = -model.w(end);
  
  svm_model = lmodel;
  return;
end

if ~isfield(m.model,'mask') | length(m.model.mask)==0
  m.model.mask = logical(ones(numel(m.model.w),1));
end

if length(m.model.mask(:)) ~= numel(m.model.w)
  m.model.mask = repmat(m.model.mask,[1 1 features]);
  m.model.mask = logical(m.model.mask(:));
end

%xsall = m.model.nsv;
%objidsall = m.model.svids;

%% look into the object inds to figure out which subset of the data
%% is actually hard negatives for mining
if mining_params.extract_negatives == 1
  [negatives,vals,pos,m] = find_set_membership(m);
  %xs = m.model.nsv(:,[negatives pos]);
  %objids = m.model.svids([negatives pos]);

  xs = m.model.nsv(:, [negatives]);
  objids = m.model.svids([negatives]);
  
  %xs_val = m.model.nsv(:,[negatives pos]);
  %objids_val = m.model.svids([vals]);
  
  %bg = get_pascal_bg('trainval');
  
  %fprintf(1,'getting overlaps with gt\n');
  %tic
  %[maxos,maxind,maxclass] = ...
  %    get_overlaps_with_gt(m, [objids{:}], bg);
  %toc
  
  %keyboard
  %VOCinit;
  %targetc = find(ismember(VOCopts.classes,m.cls));
  %gainvec = maxos + .5*double(maxclass==targetc);
  
else
  xs = m.model.nsv;
  objids = m.model.svids;
end

%maxos = cellfun(@(x)x.maxos,objids);
%maxclass = cellfun(@(x)x.maxclass,objids);
%xs = xs(:,maxos<.5);
%objids = objids(maxos<.5);

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
  objids = objids(r);
  %gainvec = gainvec(r);
end

if 0
  [aa,bb] = sort(gainvec,'descend');
  goods = find(aa>.5);
  aa = aa(goods);
  bb = bb(goods);
  xd = xs(:,bb);
  dd = diff(xd,[],2);
  superx = -dd;
  neggies = xs(:,gainvec<.2);
  supery = cat(1,ones(size(superx,2),1),...
               -1*ones(size(neggies,2),1));
  superx = cat(2,superx,neggies);
  fprintf(1,'liblinearing...\n');
  
  tic
  model = liblinear_train(supery, sparse(superx)', sprintf(['-s 2 -B -1 -c' ...
                    ' %f'],1000));%mining_params.SVMC));
  toc
  wex = model.w(1:end)';
  b = 0;
end
if 1
  %old method

  % wm = zeros(m.model.hg_size(1),m.model.hg_size(2));
  
  % wm(1:end,1) = 1;
  % wm(1:end,end) = 1;
  % wm(1,1:end) = 1;
  % wm(end,1:end) = 1;
  % wm = bwdist(wm);
  % wm = max(wm(:))-wm;
  % wm = wm / max(wm(:));
  % wm = (exp(-2*wm));
  % wm = double(repmat(wm,[1 1 features]));
  
  superx = cat(2,m.model.x,xs);
  supery = cat(1,ones(size(m.model.x,2),1),-1*ones(size(xs,2),1));


  
  if 0 %size(m.model.x,2) > 1
    %superx2 = -1*repmat(m.model.x(:,1), 1,size(m.model.nsv,2))-m.model.nsv(:,1:end);
    %supery2 = 1*ones(size(m.model.nsv,2),1);

    %% add self max
    superx2 = m.model.xstart*10;
    supery2 = 1;

    superx = cat(2,superx,superx2);
    supery = cat(1,supery,supery2);
  end
  
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
  % A = get_dominant_basis(reshape(mean(superx(:,supery==1),2), ...
  %                                m.model.hg_size),...
  %                        mining_params ...
  %                        .DOMINANT_GRADIENT_PROJECTION_K);
  
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
  
  % mA = mean(A,2);
  % mA = reshape(mA,hg_size);
  % for aa = size(mA,1)
  %   for bb = size(mA,2)
  %     curx = mA*0;
  %     curx(aa,bb,:) = mA(aa,bb,:);
  %     curx = curx(:) - mean(curx(:));
  %     curx = curx(:) / norm(curx(:));
  %     A = [A curx];
  %   end
  % end
  
  %% add some ones
  A(:,end+1) = 1;
  A(:,end) = A(:,end) / norm(A(:,end));
end

newx = bsxfun(@minus,superx,mu);
newx = newx(logical(m.model.mask),:);
newx = A(m.model.mask,:)'*newx;

fprintf(1,' -----\nStarting SVM dim=%d... s+=%d, s-=%d ',size(newx,1),spos,sneg);
starttime = tic;


svm_model = libsvmtrain(supery, newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'],mining_params.SVMC, wpos));

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
end %end old methods


m.model.w = reshape(wex,size(m.model.w));
m.model.b = b;

r = m.model.w(:)'*m.model.nsv - m.model.b;
svs = find(r >= -1.0000);

%KEEP 3#SV vectors (but at most max_negatives of them)
total_length = ceil(mining_params.beyond_nsv_multiplier*length(svs));
total_length = min(total_length,mining_params.max_negatives);

[alpha,beta] = sort(r,'descend');
svs = beta(1:min(length(beta),total_length));
m.model.nsv = m.model.nsv(:,svs);
m.model.svids = m.model.svids(svs);
%TODO: Note, we are keeping vectors, but some are actually for
%validation, some for training...


