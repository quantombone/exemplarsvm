function showM(models,M)
%Show the M matrix which, as the co-activation matrix, is a type of
%visual memex matrix

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
    %if aa(q) <= 1
    %  break
    %end
    subplot(4,4,q+1)

    flipper = 0;
    if bb(q) > length(models)
      bb(q) = bb(q) - length(models);
      flipper = 1;
      I=flip_image(get_exemplar_icon(models,bb(q)));
    else
      I=get_exemplar_icon(models,bb(q));
    end
    imagesc(I)
    axis image
    axis off
    title(sprintf('%d F=%d %.2f',bb(q),flipper,aa(q)))
  end
  
  pause
end