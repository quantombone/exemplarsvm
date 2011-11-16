%busmat
n{1}='bus';
c{1} = [.232 .352 .394; ...
        .172 .215 .316;...
        .236 .310 .397];

n{2} = 'cow';
c{2} =[.132 .155 .164;...
       .149 .166 .223;...
       .118 .147 .186];

n{3}='diningtable';
c{3}=[.093 .096 .102;...
      .007 .093 .053;...
      .016 .023 .111];
    
n{4}='motorbike';
c{4} =[.308 .310 .373;...
       .259 .269 .376;...
       .287 .320 .394];
    
n{5}='sheep';            
c{5}=[.071 .180 .217;...
      .113 .157 .232;...
      .117 .167 .226];

n{6}='train';
c{6} = [.206 .308 .337;
        .111 .117 .133;
        .205 .291 .369];

for i = 1:length(c)
  c{i} = c{i}([2 3 1],:);
end


figure(14)
clf
subplot(3,1,1)
for i = 1:length(n)
  plot([50 100 200],c{i}(:,1),'LineWidth',4)
  hold all
  
end

for i = 1:length(n)
  plot([50 100 200],c{i}(:,1),'k.','MarkerSize',24)
  hold all;
end
legend(n,'Location','SouthEast');
grid on;
xlabel('#Negatives');
ylabel('AP')
title('ESVM')

subplot(3,1,2)
for i = 1:length(n)
  plot([50 100 200],c{i}(:,2),'LineWidth',4)
  hold all;
end

for i = 1:length(n)
  plot([50 100 200],c{i}(:,2),'k.','MarkerSize',24)
  hold all;
end
grid on;
xlabel('#Negatives');
ylabel('AP')
title('ESVM+Cal')


subplot(3,1,3)
for i = 1:length(n)
  plot([200 50 100],c{i}(:,3),'LineWidth',4)
  hold all
end
for i = 1:length(n)
  plot([200 50 100],c{i}(:,3),'k.','MarkerSize',24);
end
grid on;
xlabel('HOG Template Size');
ylabel('AP')
title('ESVM+Co')

%set(gca,'XTick',[1 2 3 4])
%set(gca,'XTickLabel',{'1','2','3','4'});

set(gcf,'PaperPosition',[0 0 6 9]);
print(gcf,'-depsc2','/nfs/baikal/tmalisie/nn311/hog-sizer3.eps')