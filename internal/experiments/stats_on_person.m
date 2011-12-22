function supportnum = stats_on_person(stats)

targetcls = 'horse';
%targetcls = 'person';

supportcls = {'person'};
%supportcls = {'bicycle','motorbike','horse'};
supportnum = [0];% 0 0];

sumperson = 0;

for i = 1:length(stats.recs)
  fprintf(1,'.');
  curc = {stats.recs(i).objects.class};
  curb = cat(1,stats.recs(i).objects.bbox);
  
  bads = find([stats.recs(i).objects.difficult]);
  curc(bads) = [];
  curb(bads,:) = [];
  
  hits = find(ismember(curc,targetcls));
  cursum = length(hits);
  if cursum == 0
    continue
  end
  sumperson = sumperson + cursum;
  osmat = getosmatrix_bb(curb,curb);
  
  for j = 1:length(hits)
    for k = 1:length(supportcls)
      curhits = find(ismember(curc,supportcls{k}));
      maxos = max(osmat(hits(j),curhits),[],2);
      if length(maxos)==1 && maxos > .1
        supportnum(k) = supportnum(k)+1;
      end
    end
  end  
end

supportnum = supportnum ./ sumperson;

