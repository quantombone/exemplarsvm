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

  w = final(:,i);
  w = w - mean(w(:));

  ro = randperm(length(negs));
  %[tmp,ro] = sort(w'*final(:,negs));

  negs = negs(ro);
  negs = negs(1:min(length(negs),10000));
  
  newx = [final(:,i) final(:,poses) final(:,negs)];
  newy = [1; ones(length(poses),1); -1*ones(length(negs),1)];

  gamma = 1.8;
  SVMC = 1;
  g = double(ycat==ycat(i));
  index = 1;
  
  Iquery = (max(0.0,min(1.0,imresize(convert_to_I(ids{i}),[100 100]))));

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
  
  pak1 = mean(saveycat(bb(1:20))==ycat(i));
  

  stacker1 = cell(4,4);
  stacker1{1} = Iquery;
  
  subplot(4,4,1)
  imagesc(Iquery)
  
  title('Query')
  axis image
  axis off
  for j = 2:16
    subplot(4,4,j)

    Icur = max(0.0,min(1.0,imresize(convert_to_I(saveids{bb(j-1)}),[100 ...
                    100])));
    stacker1{j} = Icur;
    imagesc(Icur)
    title(num2str(aa(j)))
    axis image
    axis off
  end
  drawnow
  
  clear sss;
  for j = 1:size(stacker1,1),sss{j} = cat(2,stacker1{j,:}); end
  sss = cat(1,sss{:});
  stacker1 = sss;
  
  
  %print(gcf,'-dpng',sprintf('/nfs/baikal/tmalisie/labelme400/www/sunmatch/%08d.1.png',i));
  figure(2)
  w = final(:,(i)); 
  w = w - mean(w(:));
  res = w'*savefinal - b;
  [aa,bb] = sort(res,'descend');
  pak2 = mean(saveycat(bb(1:20))==ycat(i));
  stacker2 = cell(4,4);
  subplot(4,4,1)
  imagesc(Iquery)
  stacker2{1} = Iquery;
  title('Query')
  axis image
  axis off
  for j = 2:16
    subplot(4,4,j)
    
    Icur=(max(0.0,min(1.0,imresize(convert_to_I(saveids{bb(j-1)}),[100 ...
                    100]))));
    imagesc(Icur)
    stacker2{j} = Icur;
    title(num2str(aa(j)))
    axis image
    axis off
  end
  
  
  drawnow
  %print(gcf,'-dpng',sprintf('/nfs/baikal/tmalisie/labelme400/www/sunmatch/%08d.2.png',i));
  
  clear sss;
  for j = 1:size(stacker2,1),sss{j} = cat(2,stacker2{j,:}); end
  sss = cat(1,sss{:});
  stacker2 = sss;

  Ires = cat(2,pad_image(stacker1,20,1),pad_image(stacker2,20,1));


  filer = sprintf(['/nfs/baikal/tmalisie/labelme400/www/sunmatch/%s' ...
                   '%08d_%.3f_%.3f.png'],strrep(catnames{ycat(i)},'/','-'),i,pak1,pak2);
  fprintf(1,'filer is %s\n',filer);
  imwrite(Ires,filer);
  %print(gcf,'-dpng',);

end