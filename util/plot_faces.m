function plot_faces(faces)
for q = 1:length(faces)
  if length(faces{q})>0
    f = faces{q};
    hold on;
    plot([f(:,1); f(1,1)],[f(:,2); f(1,2)],'k','LineWidth', ...
         4);
  end
end
