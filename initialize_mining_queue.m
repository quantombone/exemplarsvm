function mining_queue = initialize_mining_queue(bg)
%Initialize the mining queue with 1:N ordering

fprintf(1,'Randomizing mining queue\n');
rrr = randperm(length(bg));
%rrr = 1:length(bg);

for zzz = 1:length(bg)
  mining_queue{zzz}.index = rrr(zzz);
  mining_queue{zzz}.num_visited = 0;
end
