function mining_queue = initialize_mining_queue(bg,rrr)
%Initialize the mining queue with random ordering, if ordering is
%not specified

if ~exist('rrr','var')
  fprintf(1,'Randomizing mining queue\n');
  myRandomize;
  rrr = randperm(length(bg));
  %rrr = 1:length(bg);
end

for zzz = 1:length(rrr)
  mining_queue{zzz}.index = rrr(zzz);
  mining_queue{zzz}.num_visited = 0;
end
