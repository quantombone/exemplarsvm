function nnslide
%% here we test out a NN-slide

m = load(['/nfs/baikal/tmalisie/local/VOC2007/exemplars/' ...
          '000540.1.train.mat']);
m = m.m;
VOCinit;

curid = m.curid;
Ibase = imread(sprintf(VOCopts.imgpath,curid));
I = im2double(Ibase);
imagesc(Ibase)

f = featpyramid2(I,m.model.params.sbin, 10);

%aaa = 8;
%bbb = 3;
x = f{1}(20:30,20:30,:);
[rs,t] = localizemeHOG(I, m.model.params.sbin, ...
                         {x}, {0},...
                         -100000,...
                         50,10, ...
                         1,1);

rs.score_grid{1}(1)

figure(1)
clf
imagesc(I)
plot_bbox(rs.id_grid{1}{1}.bb)

