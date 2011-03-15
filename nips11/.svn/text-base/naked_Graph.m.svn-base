function row=naked_Graph(g)

if ~exist('g','var')
  g = initialize_all_positives;
end

orderi = 1:length(g);
%myRandomize;
%rrr = randperm(length(orderi));
%orderi = orderi(rrr);

VOCinit;
results_directory = ...
    sprintf('%s/exemplars/rows/',VOCopts.localdir);

for i = 1:length(orderi)
  startid = orderi(i);

  orderj = 1:length(g{startid}.curw);
  %rrr = randperm(length(orderj));
  %orderj = orderj(rrr);
  for j = 1:length(orderj)
    startlvl = orderj(j);
    
    filer = sprintf('%s/row.%05d.%05d.%s.mat',results_directory,...
                    startid,startlvl,g{i}.cls);
    filerlock = [filer '.lock'];
    if fileexists(filer)
      row{i}{j} = load(filer);
      row{i}{j} = row{i}{j}.row;
      fprintf(1,'.');
      continue
    end
    
    if (mymkdir_dist(filerlock)==0)
      continue
    end
    
    row = driver(g,startid,startlvl);
    save(filer,'row');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end
    
  end

  
  % %figure(2)
  % %imagesc(row{i}.diffs)
  
  % figure(3)
  % clf
  % for j = 1:9
  %   subplot(3,3,j)
  %   imagesc(show_g(g,row{i}.inds(j),row{i}.lvls(j),...
  %                  row{i}.offset(j,:),row{i}.hg_size));
  %   axis image
  %   axis off
  %   title(sprintf('id=%d scores=%.3f',row{i}.inds(j), row{i}.scores(j)));
  % end
  % drawnow
  % %baser = ['/nfs/baikal/tmalisie/labelme400/www/siggraph/cowmex/'];
  % %print(gcf,'-djpeg',sprintf('%s/%05d.jpg',baser,i))

end

% keyboard
% figure(1)
% clf
% for i = 1:9
%   subplot(3,3,i)
%   imagesc(show_g(g,row{1}.inds(i),row{1}.lvls(i),...
%                  row{1}.offset(i,:),row{1}.hg_size));
% end
% drawnow

% for j = 1:10
%   row{j+1} = driver(g,row{1}.inds(j+1),row{1}.lvls(j+1));


%   figure(j+1)
%   clf
%   for i = 1:9
%     subplot(3,3,i)
%     imagesc(show_g(g,row{j+1}.inds(i),row{j+1}.lvls(i)));
    
%     imagesc(show_g(g,row{j+1}.inds(i),row{j+1}.lvls(i),...
%                    row{j+1}.offset(i,:),row{j+1}.hg_size));
%   end
%   drawnow
% end

function row = driver(g,startid,startlvl)

VOCinit;
I = imread(sprintf(VOCopts.imgpath,g{startid}.curid));
I = im2double(I);

x = g{startid}.curw{startlvl};
model = get_initial_w(x, g{startid}.cls, I);
w = model.w;
b = model.b;

%w = x;
%w = x - mean(x(:));
%w = w / (size(w,1)*size(w,2));

%w = x2w(w);
m = g{startid}.curm{startlvl};

[maxs,maxl,maxoffset,maxo,maxfeats] = ...
    neighbor_scores(g,w,b,m);

curs = maxs;
%curs(maxo<.5) = curs(maxo<.5) - 100;
[aa,bb] = sort(curs,'descend');

clear row
row.w = w;
row.b = b;

K = sum(aa>=-1.0);

row.inds = bb(1:K);
row.lvls = maxl(bb(1:K));
row.scores = aa(1:K);
row.offset = maxoffset(bb(1:K),:);
row.hg_size = size(w);
row.startid = startid;
row.startlvl = startlvl;

%diffs = sum(bsxfun(@minus,maxfeats(:,row.inds),x(:)).^2,2);
%diffs = sum(reshape(diffs,size(x)),3);
%diffs = exp(-diffs);
%row.diffs = diffs;

function model = get_initial_w(x,cls, I)

w = x;
m.model.nsv = zeros(prod(size(w)),0);
m.model.svids = [];
m.model.mask = ones(size(w,1),size(w,2));
%m.model.mask(size(m.model.mask,1)/2 : end,:) = 1;
%m.model.mask(3:end-3,3:end-3)=1;
%m.model.mask = m.model.mask*0+1;

%maximum #windows per image (per exemplar) to mine
WPI = 40;

%Levels-per-octave defines how many levels between 2x sizes in pyramid
lpo = 10;

%Image scale of negative images
scaler = 1.0;

%Maximum number of negatives to mine before SVM kicks in (this
%defines one iteration)
MAXW = 2000;
thresher = -1.05;
NITER = 10;
m.model.w = w - mean(w(:));
m.model.hg_size = size(w);
m.model.b = 0;

m.model.x = [x(:)];
%m.model.x = repmat(startx(:),1,100);
m.model.params.SVMC = .01;%.01;%.01;
m.model.params.sbin = 8;

bg = get_pascal_bg('train',sprintf('-%s',cls));

mining_queue = initialize_mining_queue(bg);
myRandomize;
r = randperm(length(mining_queue));
mining_queue = mining_queue(r);

%maximum number of image-scans during training
MAX_MINES = 10;
  
maxiter = 1;
for k = 1:maxiter
  cur_queue = mining_queue(1:MAX_MINES);
  rest_queue = mining_queue(MAX_MINES+1:end);
  [m, cur_queue,lastw,bbpics,template_diff] = ...
      mine_negatives(m, cur_queue, MAXW, thresher, ...
                     WPI, lpo, scaler, NITER, m.model.params.SVMC, ...
                     bg);
  mining_queue = cat(2,rest_queue,cur_queue);
end

if 1 %% GET WIGGLES  
  xxx = replica_hits(I, m.model.params.sbin, m.model.w, m.model.b);
  m.model.x = [m.model.x xxx];
end

maxiter = 5;
for k = 1:maxiter
  cur_queue = mining_queue(1:MAX_MINES);
  rest_queue = mining_queue(MAX_MINES+1:end);
  [m, cur_queue,lastw,bbpics,template_diff] = ...
      mine_negatives(m, cur_queue, MAXW, thresher, ...
                     WPI, lpo, scaler, NITER, m.model.params.SVMC, ...
                     bg);
  mining_queue = cat(2,rest_queue,cur_queue);
end

model = m.model;