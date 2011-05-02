function pool_train
%use the worker pool to do super fast training

BASEDIR = '/nfs/baikal/tmalisie/pool/';
filer = dir([BASEDIR '*mat']);
file = [BASEDIR filer(1).name];
m = load(file);
m = m.m;
mold = m;
subs = dir([BASEDIR '/pool/*' filer(1).name]);
length(subs)
resser = cell(length(subs),1);
for i = 1:length(subs)
  fprintf(1,'.');
  resser{i} = load([BASEDIR '/pool/' subs(i).name]);
end
xs = cellfun2(@(x)x.hn.xs{1},resser);
objids = cellfun2(@(x)x.hn.objids{1},resser);
xs = cat(2,xs{:});
objids = cat(2,objids{:});
if length(objids) == 0
  fprintf(1,'warning empty objids\n');
end

xsall = xs;
objidsall = objids;

bg = get_pascal_bg('trainval');
bgtrain = get_pascal_bg('train',['-' m.cls]);
bgval = get_pascal_bg('val',['-' m.cls]);
fg = get_pascal_bg('trainval',m.cls);

bgids = zeros(length(bg),1);
aa = find(ismember(bg,bgtrain));
bgids(aa) = 1;
aa = find(ismember(bg,bgval));
bgids(aa) = 2;
aa = find(ismember(bg,fg));
bgids(aa) = 3;

ids = cellfun(@(x)x.curid,objidsall);

%Train only with negatives

goods = ismember(ids,(find(bgids==1)));

xs = xs(:,goods);
objids = objids(goods);

maxos = cellfun(@(x)x.maxos,objids);
maxclass = cellfun(@(x)x.maxclass,objids);
%xs = xs(:,maxos<.5);
%objids = objids(maxos<.5);

%get current model's picture
r = m.model.w(:)'*xsall - m.model.b;
[aa,bb] = sort(r,'descend');
bg = get_pascal_bg('trainval');
NNN = 10;
Isv = get_sv_stack(objidsall(bb(1:NNN*NNN)), bg, m, NNN,NNN);
%figure(1)
%imagesc(Isv)
%drawnow

if size(xs,2) >= 5000
  %NOTE: random is better than top 5000
  r = randperm(size(xs,2));
  %r = m.model.w(:)'*xs;
  %[tmp,r] = sort(r,'descend');
  r = r(1:5000);
  xs = xs(:,r);
  objids = objids(r);
end

superx = cat(2,m.model.x,xs);
old_scores = m.model.w(:)'*superx - m.model.b;
supery = cat(1,ones(size(m.model.x,2),1),-1*ones(size(xs,2),1));
mining_params = get_default_mining_params;
mining_params.BALANCE_POSITIVES = 0;

m3 = [];
mining_params.SVMC = .01;


[wex,b,svm_model] = do_svm(supery, superx, mining_params, m3, ...
                           m.model.hg_size, old_scores);

m.model.w = reshape(wex,size(m.model.w));
m.model.b = b;
r = m.model.w(:)'*xsall - m.model.b;
[aa,bb] = sort(r,'descend');
bg = get_pascal_bg('trainval');
Isv2 = get_sv_stack(objidsall(bb(1:NNN*NNN)), bg, m, NNN, NNN);
figure(1)
clf
imagesc(cat(2,pad_image(Isv,10),pad_image(Isv2,10)))
drawnow

% r = m.model.w(:)'*superx-m.model.b;
% r2 = mold.model.w(:)'*superx-mold.model.b;
% xxx = 1:length(rrr);
% figure(2)
% clf
% plot(xxx,r2,'b.')
% hold on;
% drawnow

save(file,'m');