figure(1)
clf
I = ScreenCapture;
imagesc(I)
h=title('select capture region');
set(h,'FontSize',24);

[x,y] = ginput(2);
subber = round([x(1) y(1) x(2) y(2)]);
save subber.mat subber
