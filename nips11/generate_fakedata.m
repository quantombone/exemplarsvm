function [X,I] = generate_fakedata(I)
N = 10;

I = cell(N,1);
X = zeros(1984,N);
thetas = linspace(0,180,N);

for i = 1:length(thetas)
  fprintf(1,'.');
  curI = zeros(200,200,1);
  curI(30:170,95:105) = 1;
  curI = repmat(curI,[1 1 3]);
  
  curI = imrotate(curI,thetas(i));

  %curI = curI(20:end-20,20:end-20,:);
  curI = max(0.0,min(1.0,imresize(curI,[200 200])));
  I{i} = curI;
  f = features(curI,20);
  X(:,i) = f(:);
end

