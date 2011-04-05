function show_dalal_matrix(m,catnames)

%m = m - diag(diag(m));

r = randperm(length(catnames));
for i = 1:length(r)
  [aa,bb] = sort(m(r(i),:),'descend');
  for i = 1:10
    fprintf(1,'%s\n',catnames{bb(i)});
  end
  pause
  
end