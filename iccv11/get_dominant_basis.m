function A = get_dominant_basis(meanx, K)
%Get the dominant localized gradient basis for learning
%meanx: [N x M x F] HOG feature
%K: choose top K gradients
%returns:
%A: [NMF x NMF] projection matrix such that A*meanx(:) gets
%projected onto top K gradients per cell

Klist = [1:K];
A = zeros(prod(size(meanx)),0);
for aaa = 1:size(meanx,1)
  for bbb = 1:size(meanx,2)
    curx = meanx*0;
    % curvec = squeeze(meanx(aaa,bbb,:));
    % curvec = curvec / sum(curvec);
    % curvec = curvec.^4;
    % curvec = curvec / sum(curvec);
    % curx(aaa,bbb,:) = curvec;
    % A = [A curx(:)];
    % continue;
    %wmask = zeros(size(meanx,1),size(meanx,2));
    %wmask(aaa,bbb) = 1;
    %wmask = exp(-.1*double(bwdist(wmask)).^2);
    %wmask(wmask<.5) = 0;
    %wmask = double(wmask>0);
    %wmask = repmat(wmask,[1 1 size(curx,3)]);
    
    [alpha,beta] = sort(squeeze(meanx(aaa,bbb,1:27)),'descend');

    for K = Klist
      curx(aaa,bbb,beta(K)) = alpha(K) / (eps+alpha(K));    
      A = [A curx(:)];
    end
    
    %curx = meanx*0;
    %curx(aaa,bbb,1:18) = 100;
    %A = [A curx(:)];
    
  end
end
