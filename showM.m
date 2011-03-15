function showM(models,M)
%Show the M matrix which is a type of Visual Memex MAtrix

for i = 1:length(models)
  
  [aa,bb] = sort(M.w{i},'descend');
  
  figure(1)
  clf
  subplot(4,4,1)
  imagesc(get_exemplar_icon(models,i))
  axis image
  axis off
  title(sprintf('Source exemplar %d',i))
  
  for q = 1:15
    if aa(q) <= 1.0
      continue
    end
    subplot(4,4,q+1)
    imagesc(get_exemplar_icon(models,bb(q)))
    axis image
    axis off
    title(sprintf('Friend %d s=%.3f',bb(q),aa(q)))
  end
  
  pause
  
end