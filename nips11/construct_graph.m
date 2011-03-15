function construct_graph(row_real)


for i = 1:length(row_real)

  score_mat = zeros(length(row_real{i}),length(row_real));
  for j = 1:length(row_real{i})
    score_mat(j,row_real{i}{j}.inds) = row_real{i}{j}.scores+1;
  end
  
  score_mat(:,i) = 0;
    
  counts = sum((score_mat)>0,1);
  
  %scores = (mean(score_mat,1));
  scores = max(score_mat,[],1);
  scores(counts<=3) = 0;
  [aa,bb] = sort(scores,'descend');

  K = min(10,sum(aa>0));
  
  %inds = find(scores);
  row{i}.inds = bb(1:K);
  row{i}.scores = aa(1:K);
end

N = length(row);
mat = zeros(N,N);
for i = 1:length(row)
  mat(i,row{i}.inds) = row{i}.scores;
  other.node_names{i} = sprintf('%d',i);
  other.icon_string = @(i)sprintf('image="/nfs/baikal/tmalisie/cowicons/%05d.jpg"',i);
end

A = mat & mat';

%A = (mat + mat')/2;
A = A - diag(diag(A));



% newA = A*0;
% for i = 1:size(A,1)
%   [aa,bb] = sort(A(i,:),'descend');
%   newA(i,bb(1:5)) = A(i,bb(1:5));
% end
% A = newA & newA';



make_memex_graph(A,other);



