function [w,b,alphas,pos_inds] = learn_local_capacity(x,y,index,SVMC,gamma,g,m)
%maximum-capacity learning (alpha,beta) updates

pos_inds = find(y==y(index));

if ~exist('g','var')
  fprintf(1,'Creating G from euclidean distance\n');
  ds = distSqr_fast(x(:,index),x(:,y==y(index)));
  ds = ds / mean(ds(:));
  g = exp(-1.0*ds)';
end
%g = ones(size(pos_inds));

%turn all negatives on, they always stay on
%turn all positives off at start
alphas = y*0+1;
alphas(pos_inds)=0;

self_alpha = 1.0;
%turn self on (its a positive)
alphas(index) = self_alpha;

oldgoods = [];
for k = 1:20
  fprintf(1,'#');
  goods = find(alphas>0);
  if length(oldgoods) > 0
    diffinds = setdiff(goods,oldgoods);
    if length(diffinds) == 0
      fprintf(1,'returning prematurely\n');

      alphas = alphas(pos_inds);
      return;
    end
  end


  oldgoods = goods;
  frac=(sum(alphas)-sum(y==-1)) / length(pos_inds);
  %diffx = x(:,index)/100;
  %fprintf(1,'adding fatty\n');
  
  if exist('m','var')
    %take all positives now from the exemplar
    diffx = m.model.x;
  end
  %diffx = x(:,[]);
  %diffx = -bsxfun(@minus,x,x(:,index));

  diffy = ones(size(diffx,2),1);
  

  %newy = [y(goods); diffy];
  %newx =  [x(:,goods) diffx];

  newy = y(goods);
  newx = x(:,goods);
  
  sum(newy==1)
  sum(newy==-1)
  svm_model = svmtrain(newy,newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));
  
  
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  w = svm_weights';
  b = svm_model.rho;
  
  if y(goods(1))==-1
    w = w*-1;
    b = b*-1;
  end

  if 1
    %%optimize alphas 
    if 0
      %gamma term
      hinge = @(x)max(1-x,0.0);
      loss_term = hinge((w'*x(:,pos_inds)) .* y(pos_inds)');
      
      newalphas = double(loss_term < gamma*g' );
      % [aaa,bbb]=min(loss_term(2:end));
      % target=aaa/g(bbb+1);
      % fprintf(1,'target gamma is %.5f\n',target);
      
      alphas(pos_inds) = newalphas;
      
    else
      %%TOPK

      loss_term = (1-(w'*x(:,pos_inds)) .* y(pos_inds)')-10*g';
      [alpha,beta] = sort(loss_term);
      savealphas = alphas;
      alphas(pos_inds) = 1;
      K = 10;
      
      alphas(pos_inds(beta(K+1:end)))=0;
      %randkills = (rand(K,1)>.7);
      %alphas(pos_inds(randkills)) = 0;
    end
    
    %fprintf(1,'hack non-enable sellf\n');
    %turn self on
    alphas(index) = self_alpha;

  end  
end

fprintf(1,' --frac +: %.3f, raw=%d\n',frac,sum(alphas)-sum(y==-1));

alphas = alphas(pos_inds);
