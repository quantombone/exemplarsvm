function [wex,b,svm_model] = do_svm(supery,superx,mining_params,mask,hg_size,old_scores)
%Perform the SVM learning with some pre-processing such as PCA or
%dominant gradient projection

%Tomasz Malisiewicz (tomasz@cmu.edu)

if 0
  fprintf(1,'using liblinear\n');
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
  
  % model = liblinear_train(supery2, sparse(superx2)', sprintf(['-s 3 -B 1 -c' ...
  % ' %f'],mining_params.SVMC));
  % wex = model.w(1:end-1)';
  % b = -model.w(end);
  
  svm_model = model;
  return;
end

if ~exist('mask','var') | length(mask)==0
  mask = logical(ones(size(superx,1),1));
end
spos = sum(supery==1);
sneg = sum(supery==-1);

wpos = 1;
wneg = 1;

if mining_params.BALANCE_POSITIVES == 1
  wpos = 1/spos;
  wneg = 1/sneg;
  wpos = wpos / wneg;
  wneg = wneg / wneg;
end

A = eye(size(superx,1));
mu = zeros(size(superx,1),1);

if mining_params.DOMINANT_GRADIENT_PROJECTION == 1
  A = get_dominant_basis(reshape(mean(superx(:,supery==1),2), ...
                                 hg_size),...
                         mining_params ...
                         .DOMINANT_GRADIENT_PROJECTION_K);
  %A2 = get_dominant_basis(reshape(mean(superx(:,supery==-1 & old_scores'>=-1),2), ...
  %                               hg_size),...
  %                       mining_params ...
  %                       .DOMINANT_GRADIENT_PROJECTION_K);
  %A = [A A2];
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
  
newx = A'*bsxfun(@minus,superx,mu);

fprintf(1,' -----\nStarting SVM dim=%d... s+=%d, s-=%d ',size(newx,1),spos,sneg);
starttime = tic;

%while 1
  maskinds = find(mask);

  svm_model = libsvmtrain(supery, newx(mask,:)',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'],mining_params.SVMC, wpos));

  %convert support vectors to decision boundary
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  wex = svm_weights';
  b = svm_model.rho;
  
  if supery(1) == -1
    wex = wex*-1;
    b = b*-1;
  end
  
%   break;
  
%   nbads = sum(wex(:)<0);
%   nbads
%   if nbads == 0
%     break;
%   end
%   mask(maskinds(wex<0))=0;
% end


%% project back to original space
b = b + wex'*A(mask,mask)'*mu(mask);
wex = A(mask,mask)*wex;


wex = zeros(size(superx,1),1);
wex(mask) = svm_weights;

%% issue a warning if the norm is very small
if norm(wex) < .00001
  fprintf(1,'learning broke down!\n');
end

fprintf(1,'took %.3f sec\n',toc(starttime));
