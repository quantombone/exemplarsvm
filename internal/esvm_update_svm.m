function [m, other] = esvm_update_svm(m,fraction,min_cache)
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

if nargin>0 && isstr(m)
 other = m;
 return;
end
fprintf(1,'update svm called\n');
other = 'svm';
%if no inputs are specified, just return the suffix of current method
if nargin==0
  m = '-svm';
  return;
end

if ~isfield(m,'mask') || length(m.mask)==0
  mask = (ones(numel(m.w),1));
end

if length(m.mask(:)) ~= numel(m.w)
  mask = repmat(m.mask,[1 1 m.hg_size(3)]);
end

%if isfield(m,'mask') && ~islogical(m.mask)
%  m.mask = logical(m.mask);
%end

mask = logical(mask>0);

% %NOTE: MAXSIZE is the maximum number of examples we will keep in our cache
% MAXSIZE = m.params.train_max_negatives_in_cache;
% if size(m.svxs,2) > MAXSIZE
%   fprintf(1,'WARNING:reducing SVM problem problem from %d to %d\n',...
%           size(m.svxs,2),MAXSIZE);
  
%   r = m.w(:)'*m.svxs;
%   [tmp,r] = sort(r,'descend');
%   r = r(1:MAXSIZE);
  
%   %I used to think random was better
%   %r = HALFSIZE+randperm(length(r((HALFSIZE+1):end)));
%   %r = r(1:HALFSIZE);
%   %r = [r1 r];
%   m.svxs = m.svxs(:,r);
%   m.svbbs = m.svbbs(r,:);
% end

%% here we take the best positive from the set of possible positives

if exist('fraction','var') && size(m.svxs,2) >= min_cache
  %take a random fraction of positives
  r = randperm(size(m.x,2));
  r = r(1:round(length(r)*fraction));
else
  r = 1:size(m.x,2);
end

fprintf(1,'Fraction of positives is %d/%d\n',length(r),size(m.x, ...
                                                  2));
  
superx = cat(2,m.x(logical(mask),r),m.svxs(logical(mask),:));
supery = cat(1,ones(length(r),1),-1*ones(size(m.svxs,2),1));

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

%A = eye(size(superx,1));
%mu = zeros(size(superx,1),1);

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

%newx = bsxfun(@minus,superx(logical(m.mask),:),mu(logical(m.mask)));

%newx = A(m.mask,:)'*newx;
%keyboard
%newx = newx(find(m.mask),:);

fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ',...
        size(superx,1),spos,sneg);
starttime = tic;


fprintf(1,'starting svmlsq:\n');

oldw = [m.w(:)];
oldw(end+1) = -m.b;

curw = oldw([find(logical(mask)); numel(mask)+1]);

p.regularizer = eye(size(superx,1)+1)*(1/m.params.train_svm_c);
p.regularizer(end,end) = 0;
p.c = zeros(size(superx,1)+1,1);
p.NITER = m.params.train_newton_iter;

try
  [nw] = svmlsq(supery,superx, curw, p);
catch
  keyboard
end

svm_model.w = nw';
wex = nw(1:end-1);
b = nw(end)*-1;

learning_time = toc(starttime);

vals = supery'.*(wex'*superx-b);
svmobj =  (1/m.params.train_svm_c)/2*sum(wex.^2) + sum(hinge(vals));


if sneg == 0 
  %learning had no negatives
  wex = m.w;
  b = m.b;
  fprintf(1,['esvm_update_svm: WARNINGL: # of negative support vectors' ...
             ' is 0!\n']);
  fprintf(1,'reverting to old model...\n');
else
    
  %% project back to original space
  %b = b + wex'*A(m.mask,:)'*mu(m.mask);
  %wex = A(m.mask,:)*wex;
  
  wex2 = zeros(size(m.x,1),1);
  wex2(find(mask)) = wex;
  %wex2(end) = b;
  
  wex = wex2;

  %% issue a warning if the norm is very small
  if norm(wex) < .00001
    fprintf(1,'learning broke down!\n');
  end  
end

m.w = reshape(wex, size(m.w));
m.b = b;

rpos = wex(:)'*m.x - b;
rneg = wex(:)'*m.svxs - b;
maxpos = max(wex(:)'*m.x - b);
minpos = min(rpos);
maxneg = max(rneg);

fprintf(1,'\n --- Positives (Max,min) = %.3f,%.3f \n --- Negatives (Max) = %.3f\n',...
        maxpos,minpos,maxneg);
fprintf(1,'SVM iteration took %.3f sec, ',learning_time);


svs = find(rneg >= -1.0000);

fprintf(1,'Length of nsvs is %d/%d\n',length(svs),length(rneg));

% if length(svs) == 0
%   error('Something went wrong');
% end


%KEEP (nsv_multiplier * #SV) vectors, but at most max_negatives of them
%total_length = ceil(m.params.train_keep_nsv_multiplier* ...
%                    length(svs));

%keep cache-full worth of negatives
total_length = min(length(rneg),...
                   m.params.train_max_negatives_in_cache);

[alpha,beta] = sort(rneg,'descend');
svs = beta(1:total_length);
m.svxs = m.svxs(:,svs);
m.svbbs = m.svbbs(svs,:);
m.svbbs(:,end) = rneg(svs);
m.bb(:,end) = rpos;
%fprintf(1,' kept %d negatives\n',total_length);

