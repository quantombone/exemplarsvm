function [m, other] = esvm_update_svm(m)
% Perform SVM learning for a single exemplar model, we assume that
% the exemplar has a set of detections loaded in m.svxs and m.svbbs
% Durning Learning, we can apply some pre-processing such as PCA or
% dominant gradient projection
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

other = 'svm';
%if no inputs are specified, just return the suffix of current method
if nargin==0
  m = '-svm';
  return;
end

if ~isfield(m,'mask') || length(m.mask)==0
  m.mask = logical(ones(numel(m.w),1));
end

if length(m.mask(:)) ~= numel(m.w)
  m.mask = repmat(m.mask,[1 1 m.hg_size(3)]);
  m.mask = logical(m.mask(:));
end

if isfield(m,'mask') && ~islogical(m.mask)
  m.mask = logical(m.mask);
end

%xs = m.svxs;
%bbs = m.svbbs;

%NOTE: MAXSIZE is the maximum number of examples we will keep in our cache
MAXSIZE = m.params.train_max_negatives_in_cache;
if size(m.svxs,2) > MAXSIZE
  fprintf(1,'WARNING:reducing SVM problem problem from %d to %d\n',...
          size(m.svxs,2),MAXSIZE);
  
  r = m.w(:)'*m.svxs;
  [tmp,r] = sort(r,'descend');
  r = r(1:MAXSIZE);
  
  %I used to think random was better
  %r = HALFSIZE+randperm(length(r((HALFSIZE+1):end)));
  %r = r(1:HALFSIZE);
  %r = [r1 r];
  m.svxs = m.svxs(:,r);
  m.svbbs = m.svbbs(r,:);
end

%% here we take the best positive from the set of possible positives

% r = m.w(:)'*m.x-m.b;
% uhit = unique(m.bb(:,6));
% curx = [];
% superinds = zeros(length(uhit),1);;
% for j = 1:length(uhit)
%   goods = find(m.bb(:,6)==uhit(j) | m.bb(:,6)==(2*uhit(j)));
%   [aa,bb] = sort(r(goods),'descend');
%   curx(:,end+1) = m.x(:,goods(bb(1)));
% end


superx = cat(2,m.x,m.svxs);
supery = cat(1,ones(size(m.x,2),1),-1*ones(size(m.svxs,2),1));

spos = sum(supery==1);
sneg = sum(supery==-1);

wpos = m.params.train_positives_constant;
wneg = 1;

% if mining_params.BALANCE_POSITIVES == 1
%   fprintf(1,'balancing positives\n');
%   wpos = 1/spos;
%   wneg = 1/sneg;
%   wpos = wpos / wneg;
%   wneg = wneg / wneg;
% end

A = eye(size(superx,1));
mu = zeros(size(superx,1),1);

% if mining_params.DOMINANT_GRADIENT_PROJECTION == 1  
%   A = get_dominant_basis(reshape(mean(m.x(:,1),2), ...
%                                  m.hg_size),...
%                          mining_params.DOMINANT_GRADIENT_PROJECTION_K);
  
  
%   A2 = get_dominant_basis(reshape(mean(superx(:,supery==-1),2), ...
%                                   m.hg_size),...
%                           mining_params ...
%                           .DOMINANT_GRADIENT_PROJECTION_K);
%   A = [A A2];
% elseif mining_params.DO_PCA == 1
%   [A,d,mu] = mypca(superx,mining_params.PCA_K);
% elseif mining_params.A_FROM_POSITIVES == 1
%   A = [superx(:,supery==1)];
%   cursize = size(A,2);
%   for qqq = 1:cursize
%     A(:,qqq) = A(:,qqq) - mean(A(:,qqq));
%     A(:,qqq) = A(:,qqq)./ norm(A(:,qqq));
%   end
  
%   %% add some ones
%   A(:,end+1) = 1;
%   A(:,end) = A(:,end) / norm(A(:,end));
% end

newx = bsxfun(@minus,superx,mu);
newx = newx(logical(m.mask),:);
%newx = A(m.mask,:)'*newx;
%keyboard
%newx = newx(find(m.mask),:);

fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ',...
        size(newx,1),spos,sneg);
starttime = tic;



if 0
  svm_model = libsvmtrain(supery, newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'], m.params.train_svm_c, ...
                                                wpos));
  
  
  %convert support vectors to decision boundary
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1, ...
                                size(svm_model.SVs,2)),1));
  
  wex = svm_weights';
  b = svm_model.rho;
  
  
  %do this only for libsvm
  if supery(1) == -1
    wex = wex*-1;
    b = b*-1;    
  end
else
  
  % bvalue = 1;
  % subset = 1:length(supery);
  % for q = 1:1
  %   svm_model = liblineartrain(supery(subset), sparse(newx(:,subset))',sprintf(['-s 2 -B %.3f -c' ...
  %                     ' %f -w1 %.9f -q'], bvalue, m.params.train_svm_c, ...
  %                                                   wpos));
  %   % r = svm_model.w(1:end-1)*newx+svm_model.w(end);
  %   % rbad = find(r<0.5 & supery'==1);
  %   % subset = 1:length(supery);
  %   % subset(rbad) = [];
  %   % fprintf(1,'new pos length: %d\n',length(subset));
  % end
  % wex = reshape(svm_model.w(1:end-1),[],1);
  % b = -svm_model.w(end)*bvalue; 
  
  fprintf(1,'starting svmlsq:\n');
  tic
  newx2 = newx;
  newx2(end+1,:) = 1;
  oldw = [m.w(:)];
  oldw(end+1) = -m.b;

  [nw] = svmlsq(supery,newx2, ...
                (1/m.params.train_svm_c),oldw);
  
  toc

  svm_model.w = nw';
  wex = nw(1:end-1);
  b = nw(end)*-1;

end


learning_time = toc(starttime);

vals = supery'.*(wex'*superx-b);
svmobj =  (1/m.params.train_svm_c)/2*sum(wex.^2) + sum(hinge(vals));
svmobj

 % opt.iter_max_Newton = 30;
 % opt.prec = 1e-6;
 % starttime = tic;
 % [neww,newb] = primal_svm(supery,1./double(m.params.train_svm_c),...
 %                          newx',m.w(:),-m.b,opt);
 % learning_time = toc(starttime);

% rneg = neww(:)'*m.svxs+newb;
% rpos = neww(:)'*m.x+newb;
% [aa,bb] = sort(rneg,'descend');
% astar = aa(max(1,round(length(aa)*.1)));

% abad = find(neww(:)'*newx(:,supery==1)+newb < astar);
% if length(abad) > sum(supery==1)/2
%   [aa,bb] = sort(rpos,'ascend');
%   abad = bb(1:max(1,round(length(bb)/2)));  
% end

% fprintf(1,'\n ---length of removed "bad" positives is %d\n',length(abad));
% newx(:,abad) = [];
% supery(abad) = [];
% opt.iter_max_Newton = 5;
% [neww,newb] = primal_svm(supery,1./double(m.params.train_svm_c),...
%                          newx',neww,newb,opt);
%wex = neww;
%b = -newb;

if sneg == 0 %length(svm_model.sv_coef) == 0
  %learning had no negatives
  wex = m.w;
  b = m.b;
  fprintf(1,['esvm_update_svm: WARNINGL: # of negative support vectors' ...
             ' is 0!\n']);
  fprintf(1,'reverting to old model...\n');
else
    
  %% project back to original space
  b = b + wex'*A(m.mask,:)'*mu(m.mask);
  wex = A(m.mask,:)*wex;
  
  wex2 = zeros(size(superx,1),1);
  wex2(m.mask) = wex;
  
  wex = wex2;

  %% issue a warning if the norm is very small
  if norm(wex) < .00001
    fprintf(1,'learning broke down!\n');
  end  
end

maxpos = max(wex(:)'*m.x - b);
minpos = min(wex(:)'*m.x - b);
maxneg = max(wex(:)'*m.svxs - b);

fprintf(1,'\n --- Positives (Max,min) = %.3f,%.3f \n --- Negatives (Max) = %.3f\n',...
        maxpos,minpos,maxneg);
fprintf(1,'SVM iteration took %.3f sec, ',learning_time);

m.w = reshape(wex, size(m.w));
m.b = b;

r = m.w(:)'*m.svxs - m.b;
svs = find(r >= -1.0000);

fprintf(1,'Length of nsvs is %d/%d\n',length(svs),length(r));

% if length(svs) == 0
%   error('Something went wrong');
% end


%KEEP (nsv_multiplier * #SV) vectors, but at most max_negatives of them
%total_length = ceil(m.params.train_keep_nsv_multiplier* ...
%                    length(svs));

%keep cache-full worth of negatives
total_length = length(r);
total_length = min(total_length,m.params.train_max_negatives_in_cache);

[alpha,beta] = sort(r,'descend');
svs = beta(1:min(length(beta),total_length));
m.svxs = m.svxs(:,svs);
m.svbbs = m.svbbs(svs,:);
m.svbbs(:,end) = m.w(:)'*m.svxs-m.b;
m.bb(:,end) = m.w(:)'*m.x-m.b;
%fprintf(1,' kept %d negatives\n',total_length);

