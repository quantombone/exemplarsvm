function subber = initialize_screenshot
%Show a screenshot captured from the screen, then let user select
%a selection region (choose top left corner then drag to bottom right corner)

I = ScreenCapture;
subber = select_bbox_from_image(I);
%h = title('Select capture region (top left, then bot right)');
%set(h,'FontSize',24);

%[x,y] = ginput(2);
%subber = round([x(1) y(1) x(2) y(2)]);

if nargout == 0
  save subber.mat subber
end
