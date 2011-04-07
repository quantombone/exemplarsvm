function showpca(X,ids)

N = 6;
[v,d,mu] = mypca(X,N*N);

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