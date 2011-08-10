figure(2)
bb = 1:21;
segvals = 1-(left2(bb)-min(caliby))*1/(max(caliby)-min(caliby));
bbvals  = 1-(right2(bb)-min(caliby))*1/(max(caliby)-min(caliby));

segvals = round(segvals*300)/300;
bbvals = round(bbvals*300)/300;

bar([bbvals segvals],1)
legend('Bounding Box','Segment')
axis([0 22 0 1])

blanky='';
for q = 1:23
  blanks{q}=blanky;
end
set(gca,'YTick',0:.1:1)
set(gca,'XTickLabel',blanks)
set(gca,'XTick',1:21)


vnames = {...
    'building',...
    'grass',...
    'tree',...
    'cow',...
    'horse',...
    'sheep',...
    'sky',...
    'mountain',...
    'airplane',...
    'water',...
    'face',...
    'car',...
    'bike',...
    'flower',...
    'sign',...
    'bird',...
    'book',...
    'chair',...
    'road',...
    'cat',...
    'dog',...
    'body',...
    'boat'};
myvnames = vnames(setdiff(1:23,[5 8]));

for q = 1:21
  text(q-.1,-.02,myvnames(q),'Rotation',270+45,'FontSize',18,'FontWeight','Bold')
end

set(gcf, 'PaperPosition', [0 0 18 8]);
set(gca,'FontSize',3*6,'FontWeight','bold')
h = get(gca,'Title');
set(h,'FontSize',4*6)

set(gca,'box','off');

filer = '~/www/spectral/today/context/bbvsseg.eps';
print(gcf,'-depsc',filer);


return


hold on;
plot(1:21,1-(fhmax2(bb)-min(caliby))*1/(max(caliby)-min(caliby)),'g','LineWidth',4)
hold on;
plot(1:21,1-(blockmax2(bb)-min(caliby))*1/(max(caliby)-min(caliby)),'b','LineWidth',4)

hold on;

plot(1:23,1-(pbsp2(bb)-min(caliby))*1/(max(caliby)-min(caliby)),'r--','LineWidth',4)
hold on;
plot(1:23,1-(fhsp2(bb)-min(caliby))*1/(max(caliby)-min(caliby)),'g--','LineWidth',4)
hold on;
plot(1:23,1-(blocksp2(bb)-min(caliby))*1/(max(caliby)-min(caliby)),'b--','LineWidth',4)

legend('Ncut K=1:2','FH K=1:2','Block K=1:2','NCut Superpixel','FH Superpixel','Block Superpixel','Location','SouthWest');

ylabel('Overlap Score','FontSize',20)
set(gca,'XTick',1:23)
blanky='';
for q = 1:23
  blanks{q}=blanky;
end
set(gca,'YTick',0:.1:1)
set(gca,'XTickLabel',blanks)
%set(gca,'XTickLabel',obj_strings(betainds))

for q = 1:23
  text(q-.1,-.02,vnames(q),'Rotation',270+45,'FontSize',18,'FontWeight','Bold')
end

set(gcf, 'PaperPosition', 3*[0 0 6 3]);
set(gca,'FontSize',3*6,'FontWeight','bold')
h = get(gca,'Title');
set(h,'FontSize',4*6)
axis([1 23 0 1])
set(gca,'box','off');
%filer = '~/www/spectral/today/context/compareit.eps';
%print(gcf,'-depsc',filer);



if 0
'wtr'
'sky'
'book'
'grs'
'road'
'mntn'
'bldg'
'tre'
'body'
'hrs'
'car'
'flwr'
'sign'
'fac'
'plne'
'shp'
'cow'
'cat'
'bird'
'dog'
'boat'
'chr'
'bike'
end
