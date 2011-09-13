%busmat
n{1}='bus';
c{1} = [.103 .269 .24;...
       .185 .296 .348;...
       .216 .304 .384;...
       .236 .310 .397];
n{2} = 'cow';
c{2} =[.100 .122 .149;...
      .121 .156 .164;...
      .130 .148 .178;...
      .118 .147 .186];

n{3}='diningtable';
c{3}=[.001 .048 .096;...
              .005 .018 .112;...
              .012 .019 .118;...
              .016 .023 .111];
    
n{4}='motorbike';
c{4} =[.164 .232 .253;...
            .248 .284 .338;...
            .269 .302 .377;...
            .287 .320 .394];
n{5}='sheep';            
c{5}=[.098 .154 .163;...
        .053 .163 .198;...
        .074 .172 .239;...
        .117 .167 .226];

n{6}='train';
c{6} = [.101 .196 .256;...
         .124 .258 .315;...
         .184 .305 .362;...
         .205 .291 .369];

cs = cat(3,c{:});
ms = mean(cs,3);         

ms = [.102 .108 .164;
      .135 .170 .222;
      .163 .210 .281;
      .174 .233 .264];
       
figure(14)
clf
%figure(1)
%subplot(3,1,1)

plot([25 50 100 200],ms(:,1),'r.-','LineWidth',4);
hold on;
plot([25 50 100 200],ms(:,2),'g','LineWidth',4);
hold on;
plot([25 50 100 200],ms(:,3),'b--','LineWidth',4);
hold on;

plot([25 50 100 200],ms(:,1),'r.','Marker','o','MarkerSize',12);
hold on;
plot([25 50 100 200],ms(:,2),'g.','Marker','o','MarkerSize',12);
hold on;
plot([25 50 100 200],ms(:,3),'b.','Marker','o','MarkerSize',12);
hold on;


% for i = 1:length(n)
%   plot([25 100 1000 2500],c{i}(:,1),'LineWidth',4)
%   hold all
% end

% for i = 1:length(n)
%   plot([25 100 1000 2500],c{i}(:,1),'k.','MarkerSize',24);
%   hold all;
% end
%legend(n,'Location','SouthEast');
grid on;
h=xlabel('HOG Template Size');
set(h,'FontSize',18);
h=ylabel('mAP')
set(h,'FontSize',18);
legend('ESVM','ESVM+Cal','ESVM+Co','Location','SouthEast');
h=title('mAP versus HOG Template Size')
set(h,'FontSize',24);

set(gcf,'PaperPosition',[0 0 9 6]);
print(gcf,'-depsc2','/nfs/baikal/tmalisie/nn311/hogsize_plot.eps')

return
subplot(3,1,2)
for i = 1:length(n)
  plot([25 100 1000 2500],c{i}(:,2),'LineWidth',4);
  hold all;
end

for i = 1:length(n)
  plot([25 100 1000 2500],c{i}(:,2),'k.','MarkerSize',24);
  hold all;
end
grid on;
xlabel('#Negatives');
ylabel('AP')
title('ESVM+Cal')

subplot(3,1,3)
for i = 1:length(n)
  plot([25 100 1000 2500],c{i}(:,3),'LineWidth',4)
  hold all
end
for i = 1:length(n)
  plot([25 100 1000 2500],c{i}(:,3),'k.','MarkerSize',24);
end
grid on;
xlabel('#Negatives');
ylabel('AP')
title('ESVM+Co')

%set(gca,'XTick',[1 2 3 4])
%set(gca,'XTickLabel',{'1','2','3','4'});

set(gcf,'PaperPosition',[0 0 6 9]);
print(gcf,'-depsc2','/nfs/baikal/tmalisie/nn311/sizer3.eps')