function show_turs(ims)

filename = 'Batch2.csv';
%stringer = '"%s"';
%for i = 1:41
%  stringer = [stringer ',"%s"'];
%end

[data] = textread(filename, '%s','delimiter', ',');
data = reshape(data,42,[]);

N = size(data,2)
indexes = cell(N,1);
answers = zeros(N,1);
for i = 1:N
  fprintf(1,'.');
  curims = data(32:40,i);
  curims = cellfun2(@(x)x(2:end-1),curims);
  [aa,bb] = ismember(curims,ims);
  indexes{i} = bb;

  answer = data{42,i};
  answer = answer(2:end-1);
  
  if length(answer) == 0
    answer = -1;
  else
    answer = sscanf(answer,'check%d');
  end

  answers(i) = answer;
end


cmat = zeros(length(ims),length(ims));
for i= 1:length(answers)
  if answers(i) == -1
    continue
  end
  u = indexes{i}(1);
  v = indexes{i}(1+answers(i));

  cmat(u,v) = cmat(u,v) + 1;
  %cmat(u,u) = cmat(u,u) + 1;
end

%cmat = cmat ./ (eps+repmat(sum(cmat,2),1,size(cmat,2)));

sc = sum(cmat,2);
sc = find(sc);
resser = cell(size(cmat,1),1);
for i=1:length(sc)
  [aa,bb] = sort(cmat(sc(i),:),'descend');
  curims = [ ims(sc(i)) ims(bb)];
  resser{i} = pad_image(create_grid_res(curims,12,[200 200],1),20,1);
end

I = resser{1};
%I = create_grid_res(resser,1,[500 500]);
imagesc(I)
imwrite(I,'specialcow38.png');

