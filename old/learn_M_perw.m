function M = learn_M_perw(x, exids, os)

N = size(x,2);
K = size(x,1);

for i = 1:K
  curx = x(:,exids==i);
  curos = os(exids==i);
  lenvec = 1:length(curos);
  w_start = ones(K,1);
  res_raw = score_w(curx,curos,w_start);
  
  %cury = double(curos>.2);
  %cury(cury==0) = -1;
  cury = curos;
  cury(cury<.2) = 0;
  %[alpha,beta] = sort(cury,'descend');
  %newy = cury(beta);
  %newx = curx(:,beta);
  svm_model = svmtrain(cury', curx', sprintf(['-s 3 -p .2 -c' ...
                    ' %f'],1));
  
  %convert support vectors to decision boundary
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  wex = svm_weights';
  b = svm_model.rho;
  w = wex;

  %w = fminsearch(@(x)-score_w(curx,curos,x),w);
  res_new = score_w(curx,curos,w);
  fprintf(1,'res_raw, res_news = %.3f %.3f\n',res_raw,res_new);
    
  M{i}.w = w;
  M{i}.b = b;
  
  figure(1)
  subplot(1,3,1)
  plot(w,'r.')
  title('weights')
  subplot(1,3,2)
  plot(w'*curx-b,curos,'r.')
  ylabel('os')
  xlabel('returned score')
  title('Learned ordering')
  subplot(1,3,3)
  plot(w_start'*curx,curos,'r.')
  ylabel('os')
  xlabel('full score')
  title('Raw ordering')  
  pause
end


function score = score_w(curx,curos,w)
s = w'*curx;

%score = sum(s.*(curos>.5)) - sum(s.*(curos<=.5));
%return
%keyboard
[aa,bb] = sort(s,'descend');
vals = cumsum(curos(bb)>.5)./(1:length(curos));
score = mean(vals);

%g = fspecial('gaussian',[9 1],1);

%keyboard