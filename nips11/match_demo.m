function match_demo(final,ids,ycat,catnames)
%demo of matching between things

trainset = randn(size(final,2),1)>0;

savefinal = final(:,~trainset);
saveids = ids(~trainset);
saveycat = ycat(~trainset);
final = final(:,trainset);
ids = ids(trainset);
ycat = ycat(trainset);



if 0
  %compute classes from filenames
  ids2 = cellfun2(@(x)x((strfind(x,'SUN397')+8):end),ids);
  slashes = cellfun2(@(x)strfind(x,'/'),ids2);
  slashes = cellfun(@(x)x(end),slashes);
  cats = ids2;
  for i = 1:length(ids2)
    cats{i} = ids2{i}(1:slashes(i));
  end
  ucats = unique(cats);
  [aa,bb] = ismember(cats,ucats);
end

r = randperm(size(final,2));

for iii = 1:length(r)
  i = r(iii);
  negs = find(ycat~=ycat(i));
  poses = setdiff(find(ycat==ycat(i)),i);
  
  ro = randperm(length(negs));
  negs = negs(ro);
  negs = negs(1:min(length(negs),10000));
  
  newx = [final(:,i) final(:,poses) final(:,negs)];
  newy = [1; ones(length(poses),1); -1*ones(length(negs),1)];

  gamma = 1.8;
  SVMC = 1;
  g = double(ycat==ycat(i));
  index = 1;
  
  Iquery = (max(0.0,min(1.0,imresize(convert_to_I(ids{i}),[200 200]))));

  [w,b,alphas,pos_inds] = learn_local_capacity(newx,newy,index,SVMC, ...
                                               gamma,g(newy==1));

  %[w,b,alphas,pos_inds] = learn_local_rank_capacity(newx,newy,index,SVMC, ...
  %                                             gamma,g);

  %plot(w'*newx-b)

  % SVMC = .01;
  % fprintf(1,'training svm\n');
  % svm_model = svmtrain(newy,newx',sprintf(['-s 0 -t 0 -c' ...
  %                   ' %f -q'],SVMC));
  % fprintf(1,'done\n');

  % svm_weights = full(sum(svm_model.SVs .* ...
  %                        repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  % w = svm_weights';
  % b = svm_model.rho;
  
  res = w'*savefinal - b;
  fprintf(1,'Applied result to %d items\n',length(res));

  figure(1)

  [aa,bb] = sort(res,'descend');
  subplot(4,4,1)
  imagesc(Iquery)
  title('Query')
  axis image
  axis off
  for j = 2:16
    subplot(4,4,j)
    imagesc(max(0.0,min(1.0,imresize(convert_to_I(saveids{bb(j)}),[200 ...
                    200]))))
    title(num2str(aa(j)))
    axis image
    axis off
  end
  drawnow
  print(gcf,'-dpng',sprintf('/nfs/baikal/tmalisie/labelme400/www/sunmatch/%08d.1.png',i));
  figure(2)
  w = final(:,(i)); 
  w = w - mean(w(:));
  res = w'*savefinal - b;
  [aa,bb] = sort(res,'descend');
  subplot(4,4,1)
  imagesc(Iquery)
  title('Query')
  axis image
  axis off
  for j = 2:16
    subplot(4,4,j)
    
    imagesc(max(0.0,min(1.0,imresize(convert_to_I(saveids{bb(j)}),[200 ...
                    200]))))
    title(num2str(aa(j)))
    axis image
    axis off
  end
  
  
  drawnow
  print(gcf,'-dpng',sprintf('/nfs/baikal/tmalisie/labelme400/www/sunmatch/%08d.2.png',i));


end