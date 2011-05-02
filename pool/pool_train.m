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

mining_params = get_default_mining_params;
mining_params.BALANCE_POSITIVES = 0;
mining_params.SVMC = .01;


%%NOTE hardcoded to use one exemplar, not a stack of multiple
xs = cellfun2(@(x)x.hn.xs{1},resser);
objids = cellfun2(@(x)x.hn.objids{1},resser);

xs = cat(2,xs{:});
objids = cat(2,objids{:});
if length(objids) == 0
  fprintf(1,'warning empty objids\n');
end

m = add_new_detections(m,xs,objids);
NNN = 10;
bg = get_pascal_bg('trainval');
Isv = get_sv_stack(m,bg,NNN);

[m] = do_svm(m, mining_params);

%m.model.w = reshape(wex,size(m.model.w));
%m.model.b = b;
Isv2 = get_sv_stack(m,bg,NNN);
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
