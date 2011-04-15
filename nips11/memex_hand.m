function [svm_model,bg] = memex_hand(X2,y2,ids2)

% if fileexists('handdata.mat')
%   load('handdata.mat');
% else
N = 50;
bg = get_movie_bg('/nfs/hn22/tmalisie/exemplarsvm/data/faceapt.mov', ...
                  N);

N2 = 100;
bg2 = get_movie_bg('/nfs/hn22/tmalisie/exemplarsvm/data/aptbg.mov', ...
                  N2);



X = zeros(1984,N);
y = zeros(N,1);
ids = bg;
for i = 1:N
  fprintf(1,'.');
  curI = convert_to_I(bg{i});
  curI = imresize(curI,[200 200]);
  f = features(curI,20);
  X(:,i) = f(:);
  %Is{i} = HOGpicture(f);
  y(i) = 500;
end
rrr = randperm(size(X2,2));
rrr = rrr(1:5000);
X2 = X2(:,rrr);
y2 = y2(rrr);
ids2 = ids2(rrr);

if 0
X2 = zeros(1984,N2);
y2 = zeros(N2,1);
ids2 = bg2;
for i = 1:N2
  fprintf(1,'.');
  curI = convert_to_I(bg2{i});
  curI = imresize(curI,[200 200]);
  f = features(curI,20);
  X2(:,i) = f(:);
  y2(i) = 1;
end
end

targety = 500;
X = cat(2,X,X2);
y = cat(1,y,y2);
ids = cat(1,ids,ids2);

%save('handdata.mat','X','y','targety','ids');
%end

svm_model = train_all(X,y,targety,ids);

%stacker1 = create_grid_res(Is,6);
%imagesc(stacker1)