function show_graceful_confusion(cls) %models,final)
VOCinit;

pedro = load('/onega_no_backups/users/ss1/finalAllDets.mat');
filer = sprintf('%s/nips-%s-%s.mat', VOCopts.resdir, cls, ...
                    'e-svm');
esvm = load(filer);

filer = sprintf('%s/nips-%s-%s.mat', VOCopts.resdir, cls, ...
                'e-svm-vregmine');

vregmine = load(filer);

clsid = find(ismember(pedro.categories,cls));
bbs{1} = pedro.finalDets{clsid};
scores{1} = pedro.finalDets{clsid}(:,5);
imid{1} = pedro.finalDets{clsid}(:,6);

bg = get_pascal_bg('test');
bg = cellfun(@(x)get_file_id(x),bg);

oks = find(ismember(imid{1},bg));
bbs{1} = bbs{1}(oks,:);
scores{1} = scores{1}(oks);
imid{1} = imid{1}(oks);

bbs{2} = cat(1,esvm.final.final_boxes{:});
scores{2} = bbs{2}(:,end);
imid{2} = bbs{2}(:,11);

bbs{3} = cat(1,vregmine.final.final_boxes{:});
scores{3} = bbs{3}(:,end);
imid{3} = bbs{3}(:,11);

titler{1} = 'pedro';
titler{2} = 'esvm';
titler{3} = 'vregmine';

for q = 1:length(scores)
  [aa,bb] = sort(scores{q},'descend');
  bbs{q} = bbs{q}(bb,:);
  scores{q} = aa;
  imid{q} = imid{q}(bb);
end

% bbs = cat(1,final.final_boxes{:});
% [aa,bb] = sort(bbs(:,end), 'descend');
% bbs = bbs(bb,:);

Ns = [100 500];
N = Ns(end);

for q = 1:length(scores)
  hits{q} = zeros(N,1);
end

for q = 1:length(scores)
for i = 1:N
  fprintf(1,'.');
  cur = bbs{q}(i,:);
  id = imid{q}(i);
  I = convert_to_I(sprintf(VOCopts.imgpath,sprintf('%06d', ...
                                                   id)));
  
  anno = PASreadrecord(sprintf(VOCopts.annopath,sprintf('%06d', ...
                                                   id)));
  
  gtbb = cat(1, anno.objects.bbox);
  [tmp,gtc]=ismember({anno.objects.class}, VOCopts.classes);
  os = getosmatrix_bb(cur, gtbb);
  [aa,ind] = max(os);
  
  % figure(1)
  % clf
  % imagesc(I)
  % plot_bbox(cur)
  % title(sprintf('os=%.3f cls=%s',aa,VOCopts.classes{gtc(ind)}))
  % axis image
  % axis off
  % drawnow
  % print(gcf,sprintf('/onega_no_backups/users/tmalisie/bustopdets/%s-%05d.png',titler{q},i),'-dpng');

  
  if (aa > .5)
    hits{q}(i) = gtc(ind);
  else
    hits{q}(i) = 21;
  end  
end
end
names = VOCopts.classes;
names{end+1} = 'none';

figure(1)
clf
for i = 1:length(Ns)
  for q = 1:length(hits)
    crow(q,:) = hist(hits{q}(1:Ns(i)),1:21);
  end

  subplot(length(Ns),1,i)
  bar(crow')
  axis([1 22 0 Ns(i)])

  set(gca,'XTick',1:21);
  set(gca,'XTickLabel',names)
  ylabel(cls)
  title(sprintf('Confusions from top %d Detections',Ns(i)))
end

set(gcf,'PaperPosition',[0 0 10 10])

basedir = [VOCopts.wwwdir '/graceful2/'];
if ~exist(basedir,'dir')
  mkdir(basedir);
end

print(gcf,sprintf('%s/%s-%s.eps',basedir, cls, ...
                  'all'), '-depsc2');
print(gcf,sprintf('%s/%s-%s.png',basedir, cls, ...
                  'all'), '-dpng');
