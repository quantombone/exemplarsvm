function [m,other] = do_edf(m)
%% Perform Distance Function learning for a single exemplar model.  We
% assume that the exemplar has a set of detections loaded in
% m.model.svxs and m.model.svbbs. 
% Returns: model [m] with updated classifier
% If no arguments are given returns the suffix '-edf' and
% classifier type 'dfun'
% Tomasz Malisiewicz (tomasz@cmu.edu)

other = 'dfun';
%if no inputs are specified, just return the suffix of current method
if nargin==0
  m = '-edf';
  return;
end

%If no mask is specified, then we assume that every single bin
%should be used.
if ~isfield(m.model,'mask') | length(m.model.mask)==0
  m.model.mask = logical(ones(numel(m.model.w),1));
end

%If the mask is only MxNx1, where the feature are MxNxF, then we
%repmat the mask to match the features.  This is important because
%it is sometimes smarter to save a 2D mask is 1/F the size of the
%full mask.
if length(m.model.mask(:)) ~= numel(m.model.w)
  m.model.mask = repmat(m.model.mask,[1 1 features]);
  m.model.mask = logical(m.model.mask(:));
end

mining_params = m.mining_params;

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

% A trick in case we have too many detections
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
  
newx = cat(2,m.model.x,xs);
newx = bsxfun(@minus,newx,m.model.x(:,1)).^2;

newy = cat(1,ones(size(m.model.x,2),1),-1*ones(size(xs,2),1));

number_positives = sum(newy==1);
number_negatives = sum(newy==-1);

wpos = mining_params.POSITIVE_CONSTANT;
wneg = 1;

if mining_params.BALANCE_POSITIVES == 1
  fprintf(1,'balancing positives\n');
  wpos = 1/number_positives;
  wneg = 1/number_negatives;
  wpos = wpos / wneg;
  wneg = wneg / wneg;
end

newx = newx(logical(m.model.mask),:);

fprintf(1,' -----\nStarting SVM dim=%d... s+=%d, s-=%d ',...
        size(newx,1), number_positives, number_negatives);
starttime = tic;

svm_model = libsvmtrain(newy, newx',sprintf(['-s 0 -t 0 -c' ...
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
  
  %With libsvm, the first datapoint is interpreted as class +1, so if
  %we started with a negative data point, we have to flip everything.
  if newy(1) == -1
    wex = wex*-1;
    b = b*-1;    
  end
  
  wex2 = zeros(size(newx,1),1);
  wex2(m.model.mask) = wex;
  
  wex = wex2;
  
  %% issue a warning if the norm is very small
  if norm(wex) < .00001
    fprintf(1,'learning broke down!\n');
  end  
end

maxpos = max(wex'*newx - b);
fprintf(1,' --- Max positive is %.3f\n',maxpos);

fprintf(1,'took %.3f sec\n',toc(starttime));

m.model.w = reshape(wex, size(m.model.w));
m.model.b = b;

%Take top mining_params.max_negatives detections 
r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 - m.model.b;

[alpha,beta] = sort(r,'descend');
svs = beta(1:min(length(beta),mining_params.max_negatives));
m.model.svxs = m.model.svxs(:,svs);
m.model.svbbs = m.model.svbbs(svs,:);

