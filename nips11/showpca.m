function showpca(X,y,ids,catnames)

% ims = cell(36,1);
% colors = jet(36);
% for i = 1:length(ims)
%   ims{i} = zeros(200,200,3);
%   for q = 1:3
%     ims{i}(:,:,q) = colors(i,q);
%   end
% end

% g = create_grid_res(ims, 6);

  
N = 10;
[v,d,mu] = mypca(X,N*N);
U = v'*bsxfun(@minus,mu,X);
X2 = bsxfun(@plus,mu,pinv(v')*U);

differ = sum((X-X2).^2,1);


oks = find(y==111);
for i = 1:length(oks)
  index = oks(i)
  [xs,bs,Is] = get_global_wiggles(convert_to_I(ids{index}));
  
  U = v'*bsxfun(@minus,mu,xs);
  X2 = bsxfun(@plus,mu,pinv(v')*U);
  
  differ = sum((xs-X2).^2,1);
  [alpha,beta] = sort(differ);
  g = create_grid_res(Is(beta), 4);
  imagesc(g)
  drawnow
  pause

  
  
end


for i = 1:size(u,1)
  [aa,bb] = sort(abs(u(i,:)));
  counts=hist(y(bb(1:1000)),1:397);
  [alpha,beta] = sort(counts,'descend');
  fprintf(1,'first is %s\n',catnames{beta(1:10)});
  
  chunk = bb(1:8);
  [aa,bb] = sort(abs(u(i,:)),'descend');
  counts=hist(y(bb(1:1000)),1:397);
  [alpha,beta] = sort(counts,'descend');
  fprintf(1,'second is %s\n',catnames{beta(1:10)});
  
  chunk = [chunk; bb(1:8)];

  g = create_grid_res(ids(chunk), 4);
  imagesc(g)
  title(sprintf('eigenvec %d',i));
  drawnow
  pause
  
end
  


pics = cell(0,1);
for i = 1:N*N
  pics{i} = cat(2,pad_image(HOGpicture(reshape(v(:,i),[8 8 31])),10,0),...
                pad_image(HOGpicture(reshape(-v(:,i),[8 8 31])),10,0));
 
end

g = create_grid_res(pics, N);
g = g(:,:,1);
figure(1)
imagesc(g)
return;
A=v'*bsxfun(@minus,mu,X);


for i = 1:N*N
  [alpha,beta] = sort((A(i,:)),'descend');
  figure(2)
  clf  
  for j = 1:16
    subplot(4,4,j)
    imagesc(convert_to_I(ids{beta(j)}));
    axis image
    axis off
  end
  pause
  
end

keyboard