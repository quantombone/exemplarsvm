function [m] = do_rank(m,mining_params)
%Perform VisualRegression from the current objects and features
%inside m
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
  indicator = cellfun(@(x)x.set,m.model.svids);
  negatives = find(indicator == 1);
  pos = find(indicator == 3);
  
  oldm = m;
  
  for qqq = 1:5
    [aa,bb] = sort(m.model.w(:)'*oldm.model.nsv,'descend');
    m.model.svids = oldm.model.svids(bb);
    m.model.nsv = oldm.model.nsv(:,bb);

    keepers = nms_objid(m.model.svids,.2);
    fprintf(1,'after nms keepers is %d\n',length(keepers));
    m.model.svids = m.model.svids(keepers);
    m.model.nsv = m.model.nsv(:,keepers);
    
    maxos = cellfun(@(x)x.maxos,m.model.svids);
    maxind = cellfun(@(x)x.maxind,m.model.svids);
    maxclass = cellfun(@(x)x.maxclass,m.model.svids);
        
    visual = m.model.w(:)'*m.model.nsv - m.model.b;
    visual = visual - min(visual(:));
    visual = visual / max(visual(:));
    
    VOCinit;
    targetc = find(ismember(VOCopts.classes,m.cls));
    gainvec = double(maxos>.5 & maxclass==targetc);
    %gainvec = max(0.0,(maxos-.5));
    
    [aa,bb] = sort(visual,'descend');
    [alpha,beta] = sort(bb);
    sets = cellfun(@(x)x.set,m.model.svids);
    negatives = find(sets==1);
    pos = find(sets == 3);
    %gainvec(negatives) = -.5;

    gainvec(maxclass~=targetc) = gainvec(maxclass~=targetc) - .5;
    gainvec(maxos<.1) = gainvec(maxos<.1)-.2;
    %gainvec = gainvec.*(1./beta');
    %gainvec = maxos;
    
    R = (.1*(gainvec) + visual)';
    %fprintf(1,'turned off visual\n');
    %R = visual';
    
    %R = R - min(R(:));
    %R = R / max(R(:));
    %R(maxos<.1) = -1;
    %R(negatives) = -1;
    
    respos = m.model.w(:)'*m.model.nsv(:,pos) - m.model.b;
    [aa,bb] = sort(respos,'descend');
    [alpha,beta] = sort(bb);
    %weights = 1./beta;
    weights = (1./beta).^.3;
    %weights(beta>5) = 0;
    %weights(weights <= 1./(10))=0;
    
    weights = [weights negatives*0+1];
    %set weights to ONE
    %weights = weights*0+1;
    
    X = m.model.nsv(:,[pos negatives]);
    X(end+1,:) = 1;
    R = R([pos negatives]);
    
    %[aa,bb] = sort(R,'descend');
    %X = X(:,bb);
    %R = R(bb);
    
    
    %C = eye(length(bb));
    
    %cs = (1./((1:length(bb))).^.3);
    C = diag(weights);

    lambda = .01;
    W = inv(X*C'*X'+lambda*eye(size(X,1),size(X,1)))*X*C'*R;
    w = W(1:end-1);
    b = -w(end-1);

    %w = m.model.w;
    %b = m.model.b;

    %figure(1)
    %plot(W'*X,R,'r.')
    %drawnow
    % figure(2)
    % imagesc(HOGpicture(reshape(w,[8 8 31])))
    
    m.model.w = reshape(w,size(m.model.w));
    m.model.b = b;
    figure(3)
    imagesc(get_sv_stack(m,5))
    drawnow
  end 
else
  xs = m.model.nsv;
  objids = m.model.svids;
end
